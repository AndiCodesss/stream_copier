"""Wrapper around a fine-tuned ModernBERT model for trade intent classification.

This module loads a locally stored ModernBERT transformer plus a small
classification head (LayerNorm + Linear) from safetensors artifacts. Given
a transcript segment packaged as an IntentContextEnvelope, it predicts one
of seven action labels (no_action, enter_long, enter_short, trim, exit_all,
move_stop, move_to_breakeven) with per-class probabilities and per-class
confidence thresholds calibrated during training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.models.domain import ActionTag
from app.services.interpretation.intent_context import IntentContextEnvelope

# The seven labels the classifier was trained on.
_DEFAULT_LABELS: tuple[ActionTag, ...] = (
    ActionTag.no_action,
    ActionTag.enter_long,
    ActionTag.enter_short,
    ActionTag.trim,
    ActionTag.exit_all,
    ActionTag.move_stop,
    ActionTag.move_to_breakeven,
)

_NON_ACTION_LABELS = {ActionTag.no_action}
_ENTRY_LABELS = {ActionTag.enter_long, ActionTag.enter_short, ActionTag.add}
_MANAGEMENT_LABELS = {ActionTag.trim, ActionTag.exit_all, ActionTag.move_stop, ActionTag.move_to_breakeven}


@dataclass(frozen=True)
class IntentClassifierPrediction:
    """Immutable result of one classifier forward pass.

    Stores the predicted label, its confidence, the full probability
    distribution across all labels, and the per-label decision thresholds
    from training. Helper properties aggregate probabilities into action
    families (entry vs. management vs. no_action).
    """

    tag: ActionTag
    confidence: float
    probabilities: dict[ActionTag, float]
    thresholds: dict[ActionTag, float]
    model_name: str

    def probability_for(self, tag: ActionTag) -> float:
        return self.probabilities.get(tag, 0.0)

    def threshold_for(self, tag: ActionTag, *, fallback: float) -> float:
        return self.thresholds.get(tag, fallback)

    @property
    def non_action_probability(self) -> float:
        return sum(self.probabilities.get(tag, 0.0) for tag in _NON_ACTION_LABELS)

    @property
    def action_probability(self) -> float:
        return sum(
            self.probabilities.get(tag, 0.0)
            for tag in _DEFAULT_LABELS
            if tag not in _NON_ACTION_LABELS
        )

    @property
    def entry_probability(self) -> float:
        return sum(self.probabilities.get(tag, 0.0) for tag in _ENTRY_LABELS)

    @property
    def management_probability(self) -> float:
        return sum(self.probabilities.get(tag, 0.0) for tag in _MANAGEMENT_LABELS)


class ModernBertIntentClassifier:
    """Lazy-loading wrapper for the local ModernBERT intent classifier.

    The model is loaded on first use (not at import time) to avoid slow
    startup when the classifier is disabled. The load() method reads the
    metadata JSON, tokenizer, base model, and classification head from
    disk, then moves everything to the configured device (CPU or CUDA).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._artifact_dir = settings.local_intent_classifier_dir
        self._loaded = False
        self._available = False
        self._load_error: str | None = None
        self._runtime_info: dict[str, object] = {
            "enabled": settings.enable_local_intent_classifier,
            "artifact_dir": str(self._artifact_dir),
        }
        self._metadata: dict[str, Any] | None = None
        self._labels: tuple[ActionTag, ...] = _DEFAULT_LABELS
        self._thresholds: dict[ActionTag, float] = {}
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._head: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if not self._settings.enable_local_intent_classifier:
            self._load_error = "local intent classifier disabled"
            self._runtime_info["error"] = self._load_error
            return

        metadata_path = self._artifact_dir / "classifier_metadata.json"
        head_path = self._artifact_dir / "classifier_head.safetensors"
        if not metadata_path.is_file() or not head_path.is_file():
            self._load_error = f"classifier artifacts missing under {self._artifact_dir}"
            self._runtime_info["error"] = self._load_error
            return

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            labels = tuple(ActionTag(label) for label in metadata.get("labels", []))
            if not labels:
                raise RuntimeError("classifier metadata is missing labels")
            model_reference = str(metadata.get("model_path") or metadata["model_name"])

            import torch
            from safetensors.torch import load_file
            from transformers import AutoModel, AutoTokenizer

            device = _resolve_device(torch, self._settings.local_intent_classifier_device)
            tokenizer = AutoTokenizer.from_pretrained(model_reference, local_files_only=True)
            model = AutoModel.from_pretrained(model_reference, local_files_only=True)
            head = _IntentClassifierHead(hidden_size=int(metadata["hidden_size"]), num_labels=len(labels), torch_module=torch)
            head_state = load_file(str(head_path), device="cpu")
            head.load_state_dict(head_state)
            model.to(device)
            model.eval()
            head.to(device)
            head.eval()

            self._metadata = metadata
            self._labels = labels
            self._thresholds = {
                ActionTag(label): float(value)
                for label, value in (metadata.get("thresholds") or {}).items()
                if label in ActionTag._value2member_map_
            }
            self._torch = torch
            self._tokenizer = tokenizer
            self._model = model
            self._head = head
            self._device = device
            self._available = True
            self._runtime_info = {
                "enabled": True,
                "artifact_dir": str(self._artifact_dir),
                "model_name": metadata["model_name"],
                "model_path": metadata.get("model_path"),
                "artifact_version": metadata.get("artifact_version", 1),
                "device": str(device),
                "max_length": int(metadata.get("max_length", self._settings.local_intent_classifier_max_length)),
                "labels": [label.value for label in labels],
            }
        except Exception as error:
            self._available = False
            self._load_error = str(error)
            self._runtime_info["error"] = self._load_error

    def is_available(self) -> bool:
        self.load()
        return self._available

    def load_error(self) -> str | None:
        self.load()
        return self._load_error

    def runtime_info(self) -> dict[str, object]:
        self.load()
        return dict(self._runtime_info)

    def classify(self, envelope: IntentContextEnvelope) -> IntentClassifierPrediction | None:
        """Run a forward pass and return the predicted label with probabilities."""
        self.load()
        if not self._available:
            return None

        assert self._metadata is not None
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._head is not None
        assert self._device is not None

        encoded = self._tokenizer(
            envelope.render(),
            return_tensors="pt",
            truncation=True,
            max_length=int(self._metadata.get("max_length", self._settings.local_intent_classifier_max_length)),
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with self._torch.inference_mode():
            outputs = self._model(**encoded)
            embeddings = _mean_pool(self._torch, outputs.last_hidden_state, encoded["attention_mask"])
            logits = self._head(embeddings)
            probabilities = self._torch.softmax(logits, dim=-1)[0].detach().cpu().tolist()

        probability_map = {
            label: round(float(probability), 6)
            for label, probability in zip(self._labels, probabilities, strict=False)
        }
        predicted_label = max(probability_map, key=probability_map.get)
        return IntentClassifierPrediction(
            tag=predicted_label,
            confidence=probability_map[predicted_label],
            probabilities=probability_map,
            thresholds=self._thresholds,
            model_name=str(self._metadata["model_name"]),
        )

    def close(self) -> None:
        self._loaded = False
        self._available = False
        self._tokenizer = None
        self._model = None
        self._head = None
        self._torch = None
        self._device = None


def _resolve_device(torch_module: Any, configured_device: str) -> Any:
    if configured_device == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("LOCAL_INTENT_CLASSIFIER_DEVICE=cuda but torch.cuda.is_available() is false")
        return torch_module.device("cuda")
    if configured_device == "auto" and torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cpu")


def _mean_pool(torch_module: Any, hidden_state: Any, attention_mask: Any) -> Any:
    """Average token embeddings, masking out padding tokens."""
    mask = attention_mask.unsqueeze(-1).type_as(hidden_state)
    summed = (hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class _IntentClassifierHead:
    """Small classification head: LayerNorm followed by a Linear projection."""
    def __init__(self, *, hidden_size: int, num_labels: int, torch_module: Any) -> None:
        nn = torch_module.nn
        self._module = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_labels),
        )

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._module.load_state_dict(state_dict)

    def to(self, device: Any) -> None:
        self._module.to(device)

    def eval(self) -> None:
        self._module.eval()

    def __call__(self, embeddings: Any) -> Any:
        return self._module(embeddings)
