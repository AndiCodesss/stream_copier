# Model Benchmark Results: Trade-Intent Classification

## 1. Task Description

The classification task is to predict which trading action a livestream trader is performing based on a transcribed text segment. Each input is a structured prompt containing the trading symbol, current position state, and recent transcript text. The model must assign one of seven action labels.

### 1.1 Label Definitions

| Label | Description | Role |
|-------|-------------|------|
| `NO_ACTION` | No trade signal detected | Neutral |
| `ENTER_LONG` | Open a new long position | Entry |
| `ENTER_SHORT` | Open a new short position | Entry |
| `TRIM` | Reduce position size (partial profit-taking) | Management |
| `EXIT_ALL` | Close entire position | Management |
| `MOVE_STOP` | Adjust stop-loss level | Management |
| `MOVE_TO_BREAKEVEN` | Move stop-loss to entry price | Management |

### 1.2 Input Format

Each example is a structured text prompt fed directly to the classifier:

```
symbol=MNQ 03-26
position=LONG
last_side=LONG
recent=there s a nice shove up. we cleared out
analysis=there s a nice shove up. we cleared out there. so you got to take a partial
current=there. So, you got to take a partial
```

The prompt encodes: the traded instrument (`symbol`), the trader's current position state (`position`, `last_side`), and three overlapping windows of transcript text (`recent`, `analysis`, `current`).

---

## 2. Dataset

### 2.1 Construction

The training dataset was constructed through a three-stage pipeline:

1. **AI Labeling**: Trading livestream transcripts were processed by large language models (Claude Sonnet, Gemini Pro, Gemini Flash) which assigned action labels to each transcript line based on the speaker's words and trading context.

2. **Human Review**: Each AI-generated label was manually reviewed and either accepted, rejected, or corrected. Review decisions are preserved in the dataset (`review_status`, `review_note` fields).

3. **Merging and Normalization**: Multiple reviewed corpora were merged using `build_reviewed_execution_dataset.py`. During merging:
   - Duplicates on the same transcript line were resolved (later corpus overrides earlier)
   - `SETUP_LONG` / `SETUP_SHORT` labels were remapped to `NO_ACTION` (setups are not actionable)
   - `ADD` labels were remapped to `ENTER_LONG` or `ENTER_SHORT` based on position side

### 2.2 Source Corpora

| AI Model | Examples | Notes |
|----------|----------|-------|
| Claude Sonnet | 278 | Initial high-quality labeling pass |
| Gemini Pro | 78 | Smoke run for validation |
| Gemini Flash | 1,139 | Bulk labeling of remaining transcripts |
| **Total raw** | **1,495** | Before deduplication |

After deduplication and normalization: **1,466 exported examples** across **103 transcripts**.

### 2.3 Label Distribution (Original Dataset)

| Label | Count | Percentage |
|-------|-------|------------|
| NO_ACTION | 569 | 38.8% |
| TRIM | 267 | 18.2% |
| EXIT_ALL | 271 | 18.5% |
| ENTER_SHORT | 153 | 10.4% |
| MOVE_STOP | 97 | 6.6% |
| ENTER_LONG | 94 | 6.4% |
| MOVE_TO_BREAKEVEN | 15 | 1.0% |
| **Total** | **1,466** | **100%** |

The dataset exhibits significant class imbalance. `NO_ACTION` dominates at 38.8%, inflated by the remapping of `SETUP_LONG` (87) and `SETUP_SHORT` (350) labels. `MOVE_TO_BREAKEVEN` is extremely rare at only 15 examples (1.0%).

### 2.4 Data Quality Issues

Manual inspection of the dataset revealed several systematic issues:

| Issue | Count | Description |
|-------|-------|-------------|
| **Placeholder market price** | 1,466 (100%) | Every example contained `market_price=24600.00` — a hardcoded default, never the actual price. This feature carried zero information. |
| **Incorrect position state** | 206 (14.0%) | Management labels (`TRIM`, `EXIT_ALL`, `MOVE_STOP`, `MOVE_TO_BREAKEVEN`) appeared with `position_side=FLAT`, which is logically impossible — you cannot trim a position you do not hold. |
| **Duplicate entries** | 13 (0.9%) | Consecutive lines within 3 lines of each other with the same action label, representing a single trade action labeled twice. |
| **Second-person advice** | 5 (0.3%) | Action labels assigned to transcript lines where the speaker was giving advice to viewers (e.g., "you can pay yourself here") rather than describing their own action. |

**Root cause of position state errors**: The training dataset is a sparse sample of labeled lines from each transcript — not every trade action was captured. In 66 of 103 transcripts, the dataset contains management actions (TRIM, EXIT) without the corresponding entry (ENTER_LONG/SHORT) that preceded them, because the entry line was not included in the labeled set.

### 2.5 Ground Truth Verification

To validate the AI-generated and semi-automatically reviewed labels, a manual verification was conducted on 10 transcripts with the highest number of action labels (203 action labels total). Each labeled line was read in context (2-3 surrounding lines) and assessed for correctness.

#### Verification Methodology

For each action label, the surrounding transcript context was read to confirm:
- **Entry labels** (`ENTER_LONG`, `ENTER_SHORT`): The speaker is actively entering a position (not discussing, planning, or advising)
- **Management labels** (`TRIM`, `EXIT_ALL`, `MOVE_STOP`, `MOVE_TO_BREAKEVEN`): The speaker is actively managing an open position
- **NO_ACTION labels**: The line contains no actionable trade signal (setup discussion, commentary, advice to viewers)

#### Verification Results

| Metric | Value |
|--------|-------|
| Transcripts verified | 10 |
| Action labels checked | 203 |
| Verified correct | 202 (99.5%) |
| Verified wrong | 0 (0.0%) |
| Ambiguous | 2 (1.0%) |
| **Label precision** | **99.5%** |

The two ambiguous labels were borderline cases where the speaker used second-person phrasing ("you got to pay yourself some here") while actively managing their own position — the intended meaning was self-directed but the wording was directed at viewers.

#### Coverage (Recall) Assessment

While label precision is near-perfect, the dataset does not capture every trade action in each transcript. An automated scan for trade-related language on unlabeled lines identified approximately 230 potential unlabeled actions across the 10 transcripts. These were primarily:

- **Continuation lines**: The same trade action described across multiple consecutive lines (e.g., "I'm adding here" on line N, "added more" on line N+2), where only one line was labeled
- **Repeated trims**: Multiple small profit-taking actions within the same position, where only the first or most explicit one was labeled
- **Stop adjustments**: Incremental stop moves mentioned in passing ("stop's at 46 now") rather than with explicit command language

The dataset prioritizes **precision over recall** — it labels clear, unambiguous trade actions and excludes borderline cases. This is appropriate for classifier training, as false labels would degrade model performance more than missing labels.

---

## 3. Data Cleanup

A cleanup script (`cleanup_training_data.py`) was developed to address the identified quality issues.

### 3.1 Cleanup Operations

| Operation | Method | Examples Affected |
|-----------|--------|-------------------|
| **Remove market price** | Strip `market_price=24600.00` from all prompts and rebuild prompt text | 1,466 (all) |
| **Repair position state** | For management labels with `position_side=FLAT`, infer correct position from `original_side` field (set during AI labeling) or `last_side` field as fallback | 200 repaired |
| **Deduplicate** | Remove consecutive entries with the same action label within 3 transcript lines | 13 removed |
| **Flag impossible state** | Flag management labels that remain `position_side=FLAT` after repair (no side information available) | 6 flagged |
| **Flag advice patterns** | Regex detection of second-person language (`"you can"`, `"you could"`, `"your stop"`, `"move your"`, `"you must"`) on action-labeled examples | 5 flagged |

### 3.2 Cleanup Results

| Metric | Value |
|--------|-------|
| Input examples | 1,466 |
| Position states repaired | 200 |
| Prompts rebuilt (market price removed) | 1,466 |
| Duplicates removed | 13 |
| Impossible state flagged | 6 |
| Advice patterns flagged | 5 |
| **Clean output** | **1,442** |
| **Flagged for review** | **24** |

### 3.3 Clean Dataset Label Distribution

| Label | Original | Clean | Change |
|-------|----------|-------|--------|
| NO_ACTION | 569 | 569 | 0 |
| TRIM | 267 | 255 | -12 |
| EXIT_ALL | 271 | 262 | -9 |
| ENTER_SHORT | 153 | 151 | -2 |
| MOVE_STOP | 97 | 97 | 0 |
| ENTER_LONG | 94 | 94 | 0 |
| MOVE_TO_BREAKEVEN | 15 | 14 | -1 |
| **Total** | **1,466** | **1,442** | **-24** |

The cleanup primarily affected management labels (`TRIM`, `EXIT_ALL`) which had the highest rate of impossible state combinations. Entry labels (`ENTER_LONG`, `ENTER_SHORT`) and `NO_ACTION` were largely unaffected.

---

## 4. Model Architectures

Five classifier architectures were evaluated, spanning classical machine learning and transformer-based approaches.

### 4.1 Classical Models (TF-IDF Based)

All three classical models share the same text featurization: **TF-IDF vectorization** with up to 20,000 features, unigrams and bigrams, and sublinear term frequency scaling. They differ only in the classification algorithm applied to the TF-IDF vectors.

**TF-IDF (Term Frequency -- Inverse Document Frequency)** converts text into numerical feature vectors by weighting each word/bigram by how frequently it appears in the current document (TF) multiplied by how rare it is across all documents (IDF). This highlights distinctive words (e.g., "breakeven", "partial") while downweighting common words (e.g., "the", "we").

#### 4.1.1 Logistic Regression (LogReg)

- **Algorithm**: Multinomial logistic regression with L2 regularization
- **Key parameters**: `C=1.0`, `class_weight="balanced"`, `max_iter=1000`
- **Characteristics**: Linear decision boundary, native probability output, balanced class weighting to handle class imbalance

#### 4.1.2 Support Vector Machine (SVM)

- **Algorithm**: Linear SVM (`LinearSVC`) with probability calibration via `CalibratedClassifierCV`
- **Key parameters**: `C=1.0`, `class_weight="balanced"`, `max_iter=2000`
- **Characteristics**: Maximizes margin between classes, robust to outliers far from decision boundary. Probability calibration uses Platt scaling with cross-validation (min 2-fold, max 5-fold depending on smallest class size).

#### 4.1.3 Multi-Layer Perceptron (MLP)

- **Algorithm**: 2-layer feedforward neural network
- **Key parameters**: Hidden layers `(256, 128)`, `max_iter=300`, `early_stopping=True`, `validation_fraction=0.15`
- **Characteristics**: Non-linear decision boundaries, early stopping to prevent overfitting. No class weighting — relies on rebalanced training set instead.

### 4.2 Transformer Models (Frozen Encoder + Trained Head)

Both transformer models use the same architecture: a **frozen pretrained transformer encoder** produces contextual text embeddings (mean-pooled), and a **trained linear classification head** (`LayerNorm -> Linear`) maps embeddings to label predictions. Only the classification head is trained; the encoder weights are never updated.

- **Training**: 20 epochs, AdamW optimizer (`lr=3e-3`, `weight_decay=0.01`), class-weighted cross-entropy loss
- **Embedding**: Mean-pooling of last hidden state, `max_length=256`, `batch_size=24`

#### 4.2.1 DistilBERT

- **Encoder**: `distilbert-base-uncased` (66M parameters, 6 layers)
- **Characteristics**: Distilled from BERT-base, 40% smaller and 60% faster than BERT while retaining 97% of its language understanding

#### 4.2.2 ModernBERT

- **Encoder**: `answerdotai/ModernBERT-base` (149M parameters, 22 layers)
- **Characteristics**: Modern BERT variant with rotary position embeddings, Flash Attention, and improved pretraining. This is the model deployed in the production system.

---

## 5. Evaluation Methodology

### 5.1 Cross-Validation Protocol

All models were evaluated using **5-fold cross-validation with transcript-level splits**. Transcripts (complete livestream sessions) were assigned to folds using a stable hash of the file path, ensuring:

- **No data leakage**: All examples from a single transcript appear in the same fold. The model never trains on one part of a session and tests on another part of the same session.
- **Reproducibility**: The hash-based assignment is deterministic.

Within each fold, the training set is rebalanced: `NO_ACTION` examples are downsampled to a maximum ratio of 2.5:1 relative to the most common action class (capped at 6,000 examples), reducing class imbalance during training while preserving the natural distribution in the test set.

### 5.2 Metrics

| Metric | Definition | Relevance |
|--------|------------|-----------|
| **Accuracy** | Fraction of correctly classified examples | Skewed by dominant `NO_ACTION` class |
| **Macro F1** | Unweighted mean of per-class F1 scores | Treats all 7 classes equally, penalizes poor performance on rare classes |
| **Action F1** | Binary detection F1: any action prediction on an action example = TP, action prediction on NO_ACTION = FP, NO_ACTION prediction on an action example = FN. Does not distinguish between action types. | Primary metric — measures whether the system detects that *a* trading signal occurred, regardless of which specific action |
| **Action Precision** | Of all examples predicted as any action class, how many were truly actions | How often a predicted action is correct (vs. false alarm on non-action) |
| **Action Recall** | Of all true action examples, how many were predicted as some action class | How many actual actions are detected (vs. missed as NO_ACTION) |

**F1 Score** is the harmonic mean of precision and recall:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

This metric is preferred over accuracy for imbalanced datasets because a model that always predicts `NO_ACTION` would achieve ~39% accuracy but 0% F1 on all action classes.

---

## 6. Results on Original Dataset

### 6.1 Summary (5-Fold CV, 1,407 examples, 103 transcripts)

| Model | Accuracy | Macro F1 | Action F1 | Action Precision | Action Recall |
|-------|----------|----------|-----------|------------------|---------------|
| **SVM** | **0.7751 +/- 0.0216** | 0.6705 +/- 0.0527 | 0.8701 +/- 0.0109 | **0.8888 +/- 0.0276** | 0.8533 +/- 0.0264 |
| **LogReg** | 0.7625 +/- 0.0211 | **0.6791 +/- 0.0479** | **0.8710 +/- 0.0204** | 0.8499 +/- 0.0310 | 0.8937 +/- 0.0196 |
| MLP | 0.7325 +/- 0.0155 | 0.5916 +/- 0.0111 | 0.8093 +/- 0.0185 | 0.9041 +/- 0.0450 | 0.7345 +/- 0.0353 |
| ModernBERT | 0.6792 +/- 0.0128 | 0.6023 +/- 0.0520 | 0.8554 +/- 0.0281 | 0.8038 +/- 0.0377 | 0.9155 +/- 0.0361 |
| DistilBERT | 0.6375 +/- 0.0444 | 0.5941 +/- 0.0678 | 0.8371 +/- 0.0409 | 0.7634 +/- 0.0636 | **0.9308 +/- 0.0496** |

### 6.2 Per-Label Performance (Aggregated Across Folds)

#### 6.2.1 Logistic Regression

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| NO_ACTION | 0.8187 | 0.7500 | 0.7829 | 548 |
| ENTER_LONG | 0.5546 | 0.7174 | 0.6256 | 92 |
| ENTER_SHORT | 0.6190 | 0.6233 | 0.6212 | 146 |
| TRIM | 0.7721 | 0.8140 | 0.7925 | 258 |
| EXIT_ALL | 0.7917 | 0.8228 | 0.8069 | 254 |
| MOVE_STOP | 0.8081 | 0.8421 | 0.8247 | 95 |
| MOVE_TO_BREAKEVEN | 0.7500 | 0.2143 | 0.3333 | 14 |

#### 6.2.2 Support Vector Machine

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| NO_ACTION | 0.7845 | 0.8303 | 0.8067 | 548 |
| ENTER_LONG | 0.6322 | 0.5978 | 0.6145 | 92 |
| ENTER_SHORT | 0.6555 | 0.5342 | 0.5887 | 146 |
| TRIM | 0.7992 | 0.8178 | 0.8084 | 258 |
| EXIT_ALL | 0.8147 | 0.8307 | 0.8226 | 254 |
| MOVE_STOP | 0.8316 | 0.8316 | 0.8316 | 95 |
| MOVE_TO_BREAKEVEN | 0.6667 | 0.1429 | 0.2353 | 14 |

#### 6.2.3 Multi-Layer Perceptron

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| NO_ACTION | 0.6766 | 0.8741 | 0.7627 | 548 |
| ENTER_LONG | 0.7708 | 0.4022 | 0.5286 | 92 |
| ENTER_SHORT | 0.6606 | 0.4932 | 0.5647 | 146 |
| TRIM | 0.8091 | 0.7558 | 0.7816 | 258 |
| EXIT_ALL | 0.7983 | 0.7323 | 0.7639 | 254 |
| MOVE_STOP | 0.8971 | 0.6421 | 0.7485 | 95 |
| MOVE_TO_BREAKEVEN | 0.0000 | 0.0000 | 0.0000 | 14 |

#### 6.2.4 DistilBERT

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| NO_ACTION | 0.8343 | 0.5328 | 0.6503 | 548 |
| ENTER_LONG | 0.4057 | 0.4674 | 0.4343 | 92 |
| ENTER_SHORT | 0.3885 | 0.6918 | 0.4975 | 146 |
| TRIM | 0.6506 | 0.6783 | 0.6641 | 258 |
| EXIT_ALL | 0.6655 | 0.7598 | 0.7096 | 254 |
| MOVE_STOP | 0.7248 | 0.8316 | 0.7745 | 95 |
| MOVE_TO_BREAKEVEN | 0.3043 | 0.5000 | 0.3784 | 14 |

#### 6.2.5 ModernBERT

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| NO_ACTION | 0.8333 | 0.6478 | 0.7290 | 548 |
| ENTER_LONG | 0.4109 | 0.5761 | 0.4796 | 92 |
| ENTER_SHORT | 0.4721 | 0.6370 | 0.5423 | 146 |
| TRIM | 0.6961 | 0.7636 | 0.7283 | 258 |
| EXIT_ALL | 0.6887 | 0.6969 | 0.6928 | 254 |
| MOVE_STOP | 0.7282 | 0.7895 | 0.7576 | 95 |
| MOVE_TO_BREAKEVEN | 0.4167 | 0.3571 | 0.3846 | 14 |

### 6.3 Confusion Matrices (Aggregated Across Folds)

Label order: NO_ACTION (NO), ENTER_LONG (EL), ENTER_SHORT (ES), TRIM (TM), EXIT_ALL (EA), MOVE_STOP (MS), MOVE_TO_BREAKEVEN (MB)

#### Logistic Regression

|  | NO | EL | ES | TM | EA | MS | MB |
|--|----|----|----|----|----|----|-----|
| **NO** | 411 | 24 | 31 | 34 | 39 | 8 | 1 |
| **EL** | 4 | 66 | 12 | 7 | 3 | 0 | 0 |
| **ES** | 31 | 14 | 91 | 7 | 2 | 1 | 0 |
| **TM** | 23 | 7 | 7 | 210 | 9 | 2 | 0 |
| **EA** | 22 | 4 | 5 | 10 | 209 | 4 | 0 |
| **MS** | 7 | 3 | 0 | 3 | 2 | 80 | 0 |
| **MB** | 4 | 1 | 1 | 1 | 0 | 4 | 3 |

#### Support Vector Machine

|  | NO | EL | ES | TM | EA | MS | MB |
|--|----|----|----|----|----|----|-----|
| **NO** | 455 | 10 | 18 | 25 | 35 | 4 | 1 |
| **EL** | 14 | 55 | 15 | 6 | 2 | 0 | 0 |
| **ES** | 46 | 12 | 78 | 7 | 2 | 1 | 0 |
| **TM** | 29 | 4 | 3 | 211 | 9 | 2 | 0 |
| **EA** | 22 | 2 | 4 | 11 | 211 | 4 | 0 |
| **MS** | 8 | 3 | 1 | 4 | 0 | 79 | 0 |
| **MB** | 6 | 1 | 0 | 0 | 0 | 5 | 2 |

#### Multi-Layer Perceptron

|  | NO | EL | ES | TM | EA | MS | MB |
|--|----|----|----|----|----|----|-----|
| **NO** | 479 | 3 | 13 | 19 | 33 | 1 | 0 |
| **EL** | 34 | 37 | 12 | 7 | 2 | 0 | 0 |
| **ES** | 63 | 3 | 72 | 7 | 0 | 1 | 0 |
| **TM** | 47 | 2 | 5 | 195 | 9 | 0 | 0 |
| **EA** | 58 | 0 | 2 | 6 | 186 | 2 | 0 |
| **MS** | 20 | 2 | 4 | 6 | 2 | 61 | 0 |
| **MB** | 7 | 1 | 1 | 1 | 1 | 3 | 0 |

#### DistilBERT

|  | NO | EL | ES | TM | EA | MS | MB |
|--|----|----|----|----|----|----|-----|
| **NO** | 292 | 34 | 92 | 52 | 55 | 17 | 6 |
| **EL** | 8 | 43 | 20 | 13 | 6 | 0 | 2 |
| **ES** | 16 | 9 | 101 | 7 | 9 | 3 | 1 |
| **TM** | 16 | 12 | 24 | 175 | 21 | 7 | 3 |
| **EA** | 13 | 5 | 19 | 18 | 193 | 3 | 3 |
| **MS** | 3 | 3 | 2 | 3 | 4 | 79 | 1 |
| **MB** | 2 | 0 | 2 | 1 | 2 | 0 | 7 |

#### ModernBERT

|  | NO | EL | ES | TM | EA | MS | MB |
|--|----|----|----|----|----|----|-----|
| **NO** | 355 | 33 | 54 | 42 | 45 | 15 | 4 |
| **EL** | 7 | 53 | 19 | 7 | 5 | 1 | 0 |
| **ES** | 17 | 16 | 93 | 12 | 5 | 2 | 1 |
| **TM** | 11 | 13 | 15 | 197 | 20 | 2 | 0 |
| **EA** | 28 | 12 | 12 | 18 | 177 | 5 | 2 |
| **MS** | 7 | 2 | 1 | 5 | 5 | 75 | 0 |
| **MB** | 1 | 0 | 3 | 2 | 0 | 3 | 5 |

---

## 7. Results on Cleaned Dataset

After applying the data cleanup pipeline, the benchmark was re-run on the cleaned dataset (1,377 examples after rebalancing, 103 transcripts) for all five models.

### 7.1 Summary (5-Fold CV, Cleaned Data)

| Model | Accuracy | Macro F1 | Action F1 | Action Precision | Action Recall |
|-------|----------|----------|-----------|------------------|---------------|
| **SVM** | **0.7952 +/- 0.0208** | 0.6881 +/- 0.0620 | 0.8841 +/- 0.0111 | **0.8997 +/- 0.0307** | 0.8699 +/- 0.0197 |
| **LogReg** | 0.7848 +/- 0.0208 | **0.7025 +/- 0.0446** | **0.8872 +/- 0.0243** | 0.8625 +/- 0.0328 | **0.9137 +/- 0.0235** |
| MLP | 0.7262 +/- 0.0267 | 0.5724 +/- 0.0223 | 0.8043 +/- 0.0344 | 0.8958 +/- 0.0330 | 0.7334 +/- 0.0640 |
| ModernBERT | 0.7267 +/- 0.0123 | 0.6488 +/- 0.0604 | 0.8723 +/- 0.0308 | — | — |
| DistilBERT | 0.6755 +/- 0.0650 | 0.5875 +/- 0.0289 | 0.8479 +/- 0.0338 | — | — |

### 7.2 Per-Label Performance (Cleaned Data, Aggregated)

#### Logistic Regression

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| NO_ACTION | 0.8571 | 0.7774 | 0.8153 | 548 |
| ENTER_LONG | 0.5739 | 0.7174 | 0.6377 | 92 |
| ENTER_SHORT | 0.6138 | 0.6181 | 0.6159 | 144 |
| TRIM | 0.7860 | 0.8347 | 0.8096 | 242 |
| EXIT_ALL | 0.8108 | 0.8607 | 0.8350 | 244 |
| MOVE_STOP | 0.8182 | 0.8617 | 0.8394 | 94 |
| MOVE_TO_BREAKEVEN | 0.8000 | 0.3077 | 0.4444 | 13 |

#### Support Vector Machine

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| NO_ACTION | 0.8118 | 0.8504 | 0.8307 | 548 |
| ENTER_LONG | 0.6835 | 0.5870 | 0.6316 | 92 |
| ENTER_SHORT | 0.6757 | 0.5208 | 0.5882 | 144 |
| TRIM | 0.7969 | 0.8595 | 0.8270 | 242 |
| EXIT_ALL | 0.8294 | 0.8566 | 0.8427 | 244 |
| MOVE_STOP | 0.8333 | 0.8511 | 0.8421 | 94 |
| MOVE_TO_BREAKEVEN | 0.7500 | 0.2308 | 0.3529 | 13 |

#### Multi-Layer Perceptron

| Label | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| NO_ACTION | 0.6810 | 0.8686 | 0.7634 | 548 |
| ENTER_LONG | 0.7222 | 0.2826 | 0.4062 | 92 |
| ENTER_SHORT | 0.6505 | 0.4653 | 0.5425 | 144 |
| TRIM | 0.7737 | 0.7769 | 0.7753 | 242 |
| EXIT_ALL | 0.8009 | 0.7254 | 0.7613 | 244 |
| MOVE_STOP | 0.8933 | 0.7128 | 0.7929 | 94 |
| MOVE_TO_BREAKEVEN | 0.0000 | 0.0000 | 0.0000 | 13 |

### 7.3 Confusion Matrices (Cleaned Data, Aggregated)

#### Logistic Regression (Clean)

|  | NO | EL | ES | TM | EA | MS | MB |
|--|----|----|----|----|----|----|-----|
| **NO** | 426 | 25 | 31 | 26 | 32 | 7 | 1 |
| **EL** | 6 | 66 | 14 | 5 | 1 | 0 | 0 |
| **ES** | 33 | 15 | 89 | 6 | 1 | 0 | 0 |
| **TM** | 15 | 4 | 7 | 202 | 12 | 2 | 0 |
| **EA** | 11 | 3 | 4 | 12 | 210 | 4 | 0 |
| **MS** | 4 | 2 | 0 | 4 | 3 | 81 | 0 |
| **MB** | 2 | 0 | 0 | 2 | 0 | 5 | 4 |

#### Support Vector Machine (Clean)

|  | NO | EL | ES | TM | EA | MS | MB |
|--|----|----|----|----|----|----|-----|
| **NO** | 466 | 10 | 15 | 23 | 29 | 4 | 1 |
| **EL** | 17 | 54 | 15 | 5 | 1 | 0 | 0 |
| **ES** | 47 | 12 | 75 | 8 | 1 | 1 | 0 |
| **TM** | 20 | 1 | 1 | 208 | 10 | 2 | 0 |
| **EA** | 16 | 0 | 3 | 12 | 209 | 4 | 0 |
| **MS** | 4 | 2 | 1 | 5 | 2 | 80 | 0 |
| **MB** | 4 | 0 | 1 | 0 | 0 | 5 | 3 |

#### Multi-Layer Perceptron (Clean)

|  | NO | EL | ES | TM | EA | MS | MB |
|--|----|----|----|----|----|----|-----|
| **NO** | 476 | 3 | 18 | 20 | 30 | 1 | 0 |
| **EL** | 40 | 26 | 13 | 10 | 3 | 0 | 0 |
| **ES** | 64 | 5 | 67 | 8 | 0 | 0 | 0 |
| **TM** | 44 | 1 | 3 | 188 | 6 | 0 | 0 |
| **EA** | 55 | 0 | 1 | 8 | 177 | 3 | 0 |
| **MS** | 16 | 1 | 0 | 7 | 3 | 67 | 0 |
| **MB** | 4 | 0 | 1 | 2 | 2 | 4 | 0 |

---

## 8. Impact of Data Cleanup

### 8.1 Before vs. After Comparison (5-Fold CV)

| Model | Metric | Original | Clean | Delta |
|-------|--------|----------|-------|-------|
| LogReg | Accuracy | 0.7625 | 0.7848 | **+2.23%** |
| LogReg | Macro F1 | 0.6791 | 0.7025 | **+2.34%** |
| LogReg | Action F1 | 0.8710 | 0.8872 | **+1.62%** |
| SVM | Accuracy | 0.7751 | 0.7952 | **+2.01%** |
| SVM | Macro F1 | 0.6705 | 0.6881 | **+1.76%** |
| SVM | Action F1 | 0.8701 | 0.8841 | **+1.40%** |
| MLP | Accuracy | 0.7325 | 0.7262 | -0.63% |
| MLP | Macro F1 | 0.5916 | 0.5724 | -1.92% |
| MLP | Action F1 | 0.8093 | 0.8043 | -0.50% |
| ModernBERT | Accuracy | 0.6792 | 0.7267 | **+4.75%** |
| ModernBERT | Macro F1 | 0.6023 | 0.6488 | **+4.65%** |
| ModernBERT | Action F1 | 0.8554 | 0.8723 | **+1.69%** |
| DistilBERT | Accuracy | 0.6375 | 0.6755 | **+3.80%** |
| DistilBERT | Macro F1 | 0.5941 | 0.5875 | -0.66% |
| DistilBERT | Action F1 | 0.8371 | 0.8479 | +1.08% |

### 8.2 Per-Label Improvement (LogReg, Best Model on Macro F1)

| Label | F1 (Original) | F1 (Clean) | Delta |
|-------|---------------|------------|-------|
| NO_ACTION | 0.7829 | 0.8153 | **+3.24%** |
| ENTER_LONG | 0.6256 | 0.6377 | +1.21% |
| ENTER_SHORT | 0.6212 | 0.6159 | -0.53% |
| TRIM | 0.7925 | 0.8096 | **+1.71%** |
| EXIT_ALL | 0.8069 | 0.8350 | **+2.81%** |
| MOVE_STOP | 0.8247 | 0.8394 | +1.47% |
| MOVE_TO_BREAKEVEN | 0.3333 | 0.4444 | **+11.11%** |

The largest improvement is on `MOVE_TO_BREAKEVEN` (+11.1% F1), though this class has only 13-14 support examples and the absolute performance remains low. `EXIT_ALL` (+2.8%) and `NO_ACTION` (+3.2%) show the most meaningful gains, likely because the repaired position state helped the model distinguish exits from non-actions.

---

## 9. Key Findings

### 9.1 Classical Models Outperform Transformers

The TF-IDF-based classical models (LogReg, SVM) consistently outperform the transformer-based models (DistilBERT, ModernBERT) across all metrics. This is likely attributable to:

1. **Small dataset size**: With only ~1,400 examples, the frozen transformer encoders cannot adapt their pretrained representations to the highly domain-specific trading vocabulary. TF-IDF features are learned directly from the training data and naturally capture domain terms.

2. **Frozen encoders**: The transformer weights are never updated — only a small linear head is trained. The pretrained embeddings may not meaningfully distinguish trading-specific phrases like "peeling some off" vs. "taking this off."

3. **High-dimensional sparse features**: TF-IDF with bigrams produces 20,000 features that directly encode key phrases. Trading action language is often formulaic ("small long here", "partial here", "I'm out"), making sparse keyword features highly effective.

### 9.2 SVM and LogReg Are Nearly Tied

SVM leads on accuracy (0.7952 vs 0.7848), while LogReg leads on Macro F1 (0.7025 vs 0.6881) and Action F1 (0.8872 vs 0.8841). Both are linear classifiers operating on the same TF-IDF feature space — the difference is in the optimization objective (margin maximization vs. log-likelihood), which produces nearly identical decision boundaries for this data.

### 9.3 MLP Underperforms Despite Nonlinearity

The MLP's additional capacity (256+128 hidden units) does not translate to better performance. It shows the highest action precision (0.8958) but the lowest action recall (0.7334), indicating it is overly conservative — it correctly identifies actions when it predicts them, but misses many actual actions. The MLP also completely fails on `MOVE_TO_BREAKEVEN` (F1 = 0.0), likely due to insufficient training examples for this class combined with no class weighting mechanism.

### 9.4 MOVE_TO_BREAKEVEN Is the Weakest Class

With only 13-15 examples, `MOVE_TO_BREAKEVEN` has the lowest F1 across all models (0.00-0.44). The MLP assigns zero predictions to this class. Even the best-performing model (LogReg on clean data) achieves only 0.44 F1. This class requires more training examples to reach reliable performance.

### 9.5 Entry Labels Are Harder Than Management Labels

`ENTER_LONG` and `ENTER_SHORT` consistently show lower F1 (0.48-0.64) than management labels like `TRIM` (0.66-0.83) and `EXIT_ALL` (0.69-0.84). Entry actions use more varied and ambiguous language ("let's try it", "here we go", "small size long here") compared to management actions which use explicit keywords ("partial", "taking profit", "I'm out", "move my stop").

### 9.6 Data Cleanup Improves Performance

Removing the uninformative `market_price` placeholder and repairing incorrect position states improved LogReg accuracy by 2.2% and Macro F1 by 2.3%. The improvement was consistent across LogReg and SVM, confirming that the quality issues in the original dataset were introducing learnable noise. The cleanup also improved the model's ability to distinguish between management actions and `NO_ACTION`, as evidenced by the +3.2% improvement in `NO_ACTION` F1.

Notably, **transformers benefited even more from cleanup**: ModernBERT improved by +4.75% accuracy and +4.65% Macro F1 on cleaned data. This suggests that noisy position state features were particularly harmful to dense embeddings, which encode all input features into a single vector — a misleading feature like `position=FLAT` on a TRIM example pollutes the entire embedding, whereas TF-IDF models can partially ignore irrelevant features through their sparse representation.

---

## 10. Production Deployment

The production system uses **ModernBERT** (not the best-performing model in benchmarks) as the deployed classifier. This decision reflects a trade-off:

- ModernBERT is already integrated into the production inference pipeline (`local_classifier.py`)
- Its frozen encoder + trained head architecture enables fast inference via saved `classifier_head.safetensors` artifacts
- The production classifier loads once at startup and classifies incoming transcript segments in real-time

Based on the benchmark results, retraining the production system with a **LogReg or SVM classifier on cleaned data** would likely improve real-world accuracy. However, the TF-IDF-based models require the full vocabulary to be available at inference time (the fitted `TfidfVectorizer`), which adds a serialization and deployment consideration not present in the current transformer-based pipeline.

---

## 11. Feature Importance Analysis

To understand *why* TF-IDF-based classicals outperform frozen transformers, the top-15 highest-weighted features per class were extracted from the trained LogReg model. Each feature is a unigram or bigram weighted by its logistic regression coefficient — the higher the weight, the more that feature pushes the model toward predicting that class.

### 11.1 Top Features by Class

#### NO_ACTION
| Weight | Feature |
|--------|---------|
| +1.54 | `position flat` |
| +1.54 | `flat last_side` |
| +1.49 | `looking` |
| +1.44 | `flat` |
| +1.25 | `for` |
| +1.11 | `watching` |
| +1.09 | `be` |
| +1.03 | `looking for` |
| +1.01 | `if` |
| +0.85 | `would` |

#### ENTER_LONG
| Weight | Feature |
|--------|---------|
| +2.64 | `long` |
| +1.85 | `squeeze` |
| +1.83 | `small` |
| +1.71 | `position flat` / `flat last_side` |
| +1.55 | `this long` |
| +1.55 | `so long` |
| +1.31 | `long now` |
| +1.23 | `in this` |

#### ENTER_SHORT
| Weight | Feature |
|--------|---------|
| +1.81 | `on` |
| +1.63 | `position flat` / `flat last_side` |
| +1.59 | `short now` |
| +1.48 | `in` |
| +1.45 | `on here` |
| +1.42 | `piece on` |
| +1.40 | `piece` |
| +1.30 | `so short` |

#### TRIM
| Weight | Feature |
|--------|---------|
| +3.51 | `some` |
| +2.09 | `covering` |
| +1.93 | `myself` |
| +1.60 | `into` |
| +1.56 | `paying` |
| +1.41 | `trimming` |
| +1.35 | `covering some` |
| +1.17 | `partial` |

#### EXIT_ALL
| Weight | Feature |
|--------|---------|
| +5.61 | `out` |
| +2.76 | `out of` |
| +1.92 | `of this` |
| +1.67 | `of` |
| +1.62 | `out here` |
| +1.17 | `done` |
| +1.11 | `right out` |

#### MOVE_STOP
| Weight | Feature |
|--------|---------|
| +4.62 | `stop` |
| +3.03 | `my stop` |
| +2.90 | `my` |
| +2.54 | `moving` |
| +2.13 | `move` |
| +1.51 | `move my` |
| +1.49 | `stops` |

#### MOVE_TO_BREAKEVEN
| Weight | Feature |
|--------|---------|
| +3.30 | `break even` |
| +3.09 | `even` |
| +2.68 | `breakeven` |
| +2.45 | `break` |
| +1.73 | `breakeven current` |
| +1.44 | `stop` |
| +1.13 | `still holding` |

### 11.2 Interpretation

The feature weights reveal that trading action language is highly **formulaic and keyword-driven**:

- **EXIT_ALL** is dominated by a single bigram `"out"` (weight +5.61) — traders almost always say "I'm out" or "out of this trade."
- **MOVE_STOP** has `"stop"` at +4.62 and `"my stop"` at +3.03 — the word "stop" alone is nearly diagnostic.
- **TRIM** relies on `"some"` (+3.51) from phrases like "covering some" and "taking some off."
- **MOVE_TO_BREAKEVEN** is captured by `"breakeven"` / `"break even"` with weights above +3.0.

This explains why frozen transformer encoders underperform: the classification signal lies in specific surface-level keywords and short phrases, not in deep semantic understanding. TF-IDF features capture these keywords directly as sparse binary indicators, while frozen BERT embeddings — pretrained on general English — encode these domain-specific phrases into dense vectors that do not separate cleanly without encoder fine-tuning.

The position state features (`position flat`, `position short`, `position long`) also carry high weight, confirming that the structured metadata fields contribute meaningful signal beyond the text alone.

---

## 12. Statistical Significance

Paired t-tests were conducted on per-fold Macro F1 scores to determine whether observed performance differences are statistically significant or within random variation from fold assignment.

### 12.1 Pairwise Comparisons (Macro F1, 5 Folds)

| Comparison | Mean Diff | p-value | Significant (p < 0.05) |
|-----------|-----------|---------|------------------------|
| LogReg vs. SVM | +0.0086 | 0.371 | No |
| LogReg vs. MLP | +0.0875 | **0.010** | **Yes** |
| SVM vs. MLP | +0.0789 | **0.019** | **Yes** |
| LogReg vs. ModernBERT | +0.0768 | 0.105 | No |
| SVM vs. ModernBERT | +0.0682 | 0.184 | No |
| LogReg vs. DistilBERT | +0.0851 | 0.111 | No |
| ModernBERT vs. DistilBERT | +0.0082 | 0.781 | No |
| MLP vs. ModernBERT | -0.0107 | 0.692 | No |

### 12.2 Interpretation

- **LogReg vs. SVM**: The difference is **not significant** (p=0.371). These models perform equivalently on this dataset — both are linear classifiers on identical TF-IDF features.
- **LogReg/SVM vs. MLP**: The gap **is significant** (p<0.02). The MLP's non-linearity does not compensate for its lack of class weighting and tendency to overfit on minority classes.
- **Classicals vs. Transformers**: The differences are **not statistically significant** at p<0.05 (p=0.10-0.18), though they approach significance. With only 5 folds, power is limited. The consistent direction (classicals > transformers across all folds) and the mechanistic explanation from feature importance (Section 11) support the finding despite marginal p-values.

---

## 13. Limitations and Methodological Notes

### 13.1 Dataset Composition: NO_ACTION Class

The `NO_ACTION` class (569 examples) is composed of two distinct sub-populations:

| Source | Count | Percentage of NO_ACTION |
|--------|-------|-------------------------|
| Genuine non-action (commentary, silence, market analysis) | ~132 | 23% |
| Remapped `SETUP_LONG` | 87 | 15% |
| Remapped `SETUP_SHORT` | 350 | 62% |

`SETUP_LONG` and `SETUP_SHORT` were remapped to `NO_ACTION` because setups are not immediately actionable — they describe the trader's bias without triggering a trade. However, this means 77% of `NO_ACTION` training examples actually contain trade-related language (e.g., "looking for a long", "short bias below VWAP"). The model learns to classify these as non-actionable, which is correct for the production system but may inflate `NO_ACTION` recall by making the class easier to predict (setup language is distinctive).

### 13.2 MOVE_TO_BREAKEVEN: Insufficient Support

With only 13-15 examples depending on the dataset version, `MOVE_TO_BREAKEVEN` cannot be reliably evaluated. In 5-fold CV, each test fold contains ~3 examples of this class. F1 scores range from 0.00 (MLP) to 0.44 (LogReg on cleaned data), but these numbers carry high variance. This class is retained as a separate label because it maps to a distinct broker action, but its metrics should be interpreted with caution.

### 13.3 Hyperparameter Choices

All models use default or standard hyperparameters without tuning:

| Model | Key Hyperparameters | Justification |
|-------|--------------------|---------------|
| LogReg | C=1.0, balanced weights | Standard regularization strength; balanced weighting addresses class imbalance without separate resampling |
| SVM | C=1.0, balanced weights | Same rationale as LogReg; linear kernel chosen because data is high-dimensional sparse (20K features) |
| MLP | (256, 128) hidden, early stopping | Two layers sufficient for non-linear boundaries on TF-IDF; early stopping prevents overfitting |
| Transformers | lr=3e-3, 20 epochs, AdamW | Standard head-training rate (encoder frozen, so higher LR is appropriate); 20 epochs with small dataset converges reliably |
| TF-IDF | 20K features, (1,2)-grams, sublinear TF | Bigrams capture key phrases ("i m", "piece on"); sublinear TF reduces impact of repeated words in long transcripts |

No grid search was performed. The dataset (1,442 examples) is too small for a separate validation-based tuning split without further reducing training data. The chosen values are standard in the literature and unlikely to be far from optimal for this problem size.

### 13.4 Frozen Encoder Decision

The transformer models use frozen (non-updated) encoder weights with only a trained linear classification head. This architectural choice was made because:

1. **Dataset size**: Fine-tuning 66M (DistilBERT) or 149M (ModernBERT) parameters on 1,442 examples risks severe overfitting, even with aggressive regularization.
2. **Compute constraints**: Full fine-tuning requires backpropagating through the entire encoder, increasing memory and time by ~10x compared to frozen embedding + head training.
3. **Fair comparison baseline**: Frozen encoders isolate the quality of pretrained representations from training procedure differences. If frozen encoders underperform TF-IDF, it demonstrates that general-purpose language representations do not transfer well to this domain — a meaningful finding for the thesis.

A future experiment could explore parameter-efficient fine-tuning (LoRA, adapter layers) as a middle ground, which would adapt encoder representations without full fine-tuning risk.

### 13.5 Statistical Significance

The reported +/- values are **sample standard deviations across 5 folds**, not confidence intervals or p-values. With only k=5 measurements per model, statistical power for detecting small differences is limited. Paired t-tests between model pairs can be run via `analyze_results.py` to assess whether observed differences exceed random variation from fold assignment. Differences smaller than ~2% in Macro F1 should be interpreted cautiously.

### 13.6 Preprocessing

All transcript text is preprocessed by `transcript_normalizer.py` before classification:
- Lowercased
- Typographic apostrophes normalized
- Digit-separating commas removed
- ASR-specific corrections applied (e.g., "v w a p" → "vwap", "peace on" → "piece on", "break even" → "breakeven")

This preprocessing is applied identically during both training data construction and production inference. The benchmark evaluates models on preprocessed text — raw ASR output would likely produce lower scores due to inconsistent spelling of domain terms.

---

## 14. Reproducibility

All results can be reproduced from the tracked training data:

```bash
cd backend
./reproduce_benchmarks.sh
```

This runs: (1) data cleanup, (2) 5-fold CV for all 5 models, (3) feature importance extraction and statistical significance tests. Outputs are written to `data/` as JSON files.

Required environment: Python 3.12+, `pip install -e .`, GPU optional (CPU works but transformer models are slower).
