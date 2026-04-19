from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from app.models.domain import ActionTag, EventType, ExecutionMode, StreamSession, TradeIntent, TradeSide, TranscriptSegment
from app.services.interpretation.action_language import detect_present_trade_signal, detect_setup_signal
from app.services.interpretation.candidate_detector import (
    CandidateAssessment,
    assess_trade_candidate,
    looks_candidate_continuation,
    looks_candidate_seed,
)
from app.services.interpretation.embedding_gate import EmbeddingGate
from app.services.interpretation.gemini_fallback import GeminiConfirmation, GeminiFallbackInterpreter
from app.services.interpretation.intent_context import IntentContextEnvelope
from app.services.interpretation.local_classifier import IntentClassifierPrediction, ModernBertIntentClassifier
from app.services.interpretation.transcript_normalizer import apply_trading_asr_corrections


ENTRY_SEED_PATTERNS = [
    r"\b(?:putting|put|got|have|in)\s+(?:a\s+)?(?:little|small)?\s*(?:piece|peace)\s+on\b",
    r"\b(?:putting|put|got|get|have)\s+something\s+on(?:\s+(?:here|there|now))?(?:\s+(?:just\s+)?in\s+case)?\b",
    r"\ba\s+(?:little|small)\s+(?:piece|peace)\s+on\b",
    r"\b(?:piece|peace)\s+on here\b",
    r"\bfeeler on\b",
]
DIRECT_LONG_PATTERNS = [
    r"\bi m long\b(?!\s+bias(?:ed)?)",
    r"\bi am long\b(?!\s+bias(?:ed)?)",
    r"\bi m in long\b",
    r"\bi am in long\b",
    r"\bi m in this long\b",
    r"\bi am in this long\b",
    r"\bin long now\b",
    r"\blong now through\b",
    r"\blong here\b",
    r"\bbuying here\b",
    r"\bgetting long\b",
    r"\blonging this\b",
    r"\bi m long again\b",
    r"\bi will get long again\b",
]
ENTRY_STRONG_LONG_PATTERNS = [pattern for pattern in DIRECT_LONG_PATTERNS if pattern != r"\blong here\b"]
DIRECT_SHORT_PATTERNS = [
    r"\bi m short\b(?!\s+(?:bias(?:ed)?|because\b))",
    r"\bi am short\b(?!\s+(?:bias(?:ed)?|because\b))",
    r"\bi m in short\b",
    r"\bi am in short\b",
    r"\bi m in this short\b",
    r"\bi am in this short\b",
    r"\bin short there\b",
    r"\bshort on this\b",
    r"\bshort here\b",
    r"\bselling here\b",
    r"\bgetting short\b",
]
ENTRY_STRONG_SHORT_PATTERNS = [pattern for pattern in DIRECT_SHORT_PATTERNS if pattern != r"\bshort here\b"]
STRONG_SELF_LONG_ENTRY_PATTERNS = [
    r"\bi m long\b(?!\s+bias(?:ed)?)",
    r"\bi am long\b(?!\s+bias(?:ed)?)",
    r"\bi m in long\b",
    r"\bi am in long\b",
    r"\bi m in this long\b",
    r"\bi am in this long\b",
]
STRONG_SELF_SHORT_ENTRY_PATTERNS = [
    r"\bi m short\b(?!\s+(?:bias(?:ed)?|because\b))",
    r"\bi am short\b(?!\s+(?:bias(?:ed)?|because\b))",
    r"\bi m in short\b",
    r"\bi am in short\b",
    r"\bi m in this short\b",
    r"\bi am in this short\b",
]
STRONG_ENTRY_RISK_TOKENS = ("versus", "stop", "risk", "risking", "under", "over", "below", "above", "reclaim")
ENTRY_CONTINUATION_PREFIX_PATTERNS = [
    # Conjunctions
    r"^(?:and|or|but|so|then|because)\b",
    r"^(?:if|if you|if you re|if you are|if we|if they)\b",
    # Positional context
    r"^(?:here|versus|at|from|through|under|over|stop)\b",
    r"^(?:this (?:long|short))\b",
    # Exit completions
    r"^(?:out|flat|done)\b",
    r"^(?:of this|of that|of it)\b",
    # Trim completions
    r"^(?:myself|some|more|off|partial|partials|profit)\b",
    r"^(?:a little|a piece)\b",
    # Breakeven / stop completions
    r"^(?:break ?even|even|be)\b",
    r"^(?:my stop|to be|to break ?even)\b",
]
ENTRY_INCOMPLETE_SUFFIX_PATTERNS = [
    # Conjunctions / pronouns
    r"\b(?:and|or|but|so|then|because)$",
    r"\b(?:if|if you|if you re|if you are|if we|if they)$",
    r"\b(?:i m|i am|we re|we are)$",
    # Entry-related
    r"\b(?:long|short)$",
    r"\b(?:in long|in short)$",
    r"\b(?:in this|in)$",
    r"\b(?:piece|peace)$",
    # Exit-related
    r"\b(?:all|fully)$",
    r"\b(?:out of|out)$",
    r"\b(?:stopped|knocked)$",
    r"\b(?:cut|cutting|flatten|flattening)$",
    # Trim-related
    r"\b(?:taking|peeling|trimming|covering|selling|paying)$",
    # Stop / breakeven-related
    r"\b(?:move|moving|moved)$",
    r"\b(?:my stop|stop|stops)$",
    r"\b(?:break)$",
]
ENTRY_LEADIN_PATTERN = (
    r"^(?:(?:all\s+right|alright|okay|ok|right|so|yes|yeah|yep|yup|now|well|again|then|mate|folks|guys|"
    r"let\s+s\s+see(?:\s+then)?)\s+)*"
)
ENTRY_BARE_CONTEXT_PATTERN = r"\b(?:versus|at|from|through|under|over|stop|reclaim|risk|risking|break)\b"
ENTRY_CONTEXTUAL_POSITION_NEGATIVE_PATTERNS = [
    r"\bif you(?: re| are)? in this (?:long|short)\b",
    r"\byou(?: re| are)? in this (?:long|short)\b",
    r"\bstill in this (?:long|short)\b",
    r"\bmight still be in this (?:long|short)\b",
    r"\b(?:some\s+)?of you might still be in this (?:long|short)\b",
    r"\bnot in this (?:long|short)\b",
    r"\bwish i was in this (?:long|short)\b",
    r"\bbeen in this (?:long|short)\b",
    r"\bmight be in this (?:long|short)\b",
    r"\bcould be in this (?:long|short)\b",
]
ENTRY_CONTEXTUAL_ACTION_HINTS = (
    "small size",
    "smaller size",
    "ideal entry",
    "wanted to get",
    "wanted to get something on",
    "get something on",
    "got something on",
    "one lot",
    "now",
    "again",
)
SETUP_LONG_PATTERNS = [r"\blooking for (?:the |a )?long\b", r"\bif we pull back.*long\b"]
SETUP_SHORT_PATTERNS = [r"\blooking for (?:the |a )?short\b", r"\bif we pop.*short\b"]
EXIT_PATTERNS = [
    r"\bi m out\b(?!\s+(?:here\b|of here\b|there\b))",
    r"\bi am out\b(?!\s+(?:here\b|of here\b|there\b))",
    r"\bout of this now\b",
    r"\bcut that\b",
    r"\btaking that off\b",
    r"\bstopped out\b",
    r"\ball out\b",
    r"\bflatten(?:ing)?(?: this)?\b",
    r"\bgetting flat\b",
    r"\bi m flat\b",
    r"\bi am flat\b",
]
CONDITIONAL_EXIT_PATTERNS = [
    r"\bif\b[^.]{0,160}\bi m out\b",
    r"\bif\b[^.]{0,160}\bi am out\b",
    r"\bor i m out\b",
    r"\bor i am out\b",
    r"\b(?:doesn t|does not|don t|do not)\b[^.]{0,160}\bi m out\b",
    r"\b(?:doesn t|does not|don t|do not)\b[^.]{0,160}\bi am out\b",
]
NON_ACTIONABLE_EXIT_PATTERNS = [
    r"\bi ll read (?:the )?last few questions i m out\b",
    r"\balmost stopped out\b",
    r"\bstopped out on the runner\b",
    r"\bi was in a (?:long|short) earlier\b.*\bstopped out\b",
    r"\bi stopped out\b.*\bi was like\b",
    r"\bstopped out on what i was looking\b",
    r"\bstopped out\b.*\bif you had another go\b",
    r"\bget out of this now you can\b",
    r"\bchase back in\b.*\bget stopped out\b",
    r"\bare more likely to .*get stopped out\b",
    r"\b(?:might|may|could|can|going to)\s+get stopped out\b",
    r"\b(?:am i|would i)\s+going to get stopped out\b",
    r"\bwatch myself get stopped out\b",
    r"\b(?:we|they|buyers|sellers|shorts|longs)\s+(?:get|got|are getting|re getting)\s+stopped out\b",
    r"\b(?:buyer|seller|short seller|long seller)\b.*\bgot stopped out\b",
    r"\beveryone got stopped out\b",
    r"\bfeel like you got stopped out\b",
    r"\b[a-z0-9_]+\s+saying\b.*\bi m out\b",
    r"\bi m out of bullets\b",
    r"\bi am out of bullets\b",
    r"\bstopped out\b.*\bi was looking for\b",
    r"\bstopped out\b.*\ball right if you think(?: about this)?\b",
    r"\bactually got stopped out\b.*\bso i was\b",
    r"\bstopped out\b.*\bin profit\b.*\bbut i was\b",
    r"\bi m flat(?: now)?\b.*\bi was (?:long|short)\b",
    r"\bi m out\b.*\bi was down\b",
    r"\ball out now [a-z0-9_]+\b",
    r"\byou stopped out\b",
    r"\bstopped out on the runner\b.*\bit s all right\b",
]
NON_ACTIONABLE_TRADE_PATTERNS = [
    r"\b(?:still\s+)?not paying myself\b",
    r"\bhow s me being able to pay myself\b",
    r"\bthere might be a (?:long|short) here\b",
    r"\bif they haven t stopped out\b",
    r"\bone break ?even(?: or|,? and)?\b",
    r"\bi took the (?:long|short)\b.*\b(?:made money|to here)\b",
    r"\bi was like (?:long|short) here\b",
    r"\bi was positioned (?:long|short) here\b",
    r"\bi was getting (?:long|short)\b",
    r"\bi was long again\b",
    r"\bwe were (?:long|short) here\b",
    r"\bif they are (?:long|short) here\b",
    r"\bif i was you\b",
    r"\bwould i take a (?:long|short) here\b",
    r"\bwould i be (?:long|short) here\b",
    r"\bthink about (?:buying|selling) here\b",
    r"\bthink about (?:getting )?(?:long|short)\b",
    r"\byou could argue there was a (?:long|short) here\b",
    r"\byou could argue there s a (?:long|short) here\b",
    r"\bwouldn t be adding here\b",
    r"\byou just want me to say\b.*\bi m (?:buying|selling) here\b",
    r"\bif you took the (?:long|short)\b",
    r"\bi said you could be (?:long|short) here\b",
    r"\bwould i consider a (?:long|short)\b",
    r"\byou could(?: potentially)? play a (?:long|short)\b",
    r"\byou could think about getting (?:long|short)\b",
    r"\byou can think about getting (?:long|short)\b",
    r"\byou could think about (?:a )?(?:long|short)\b",
    r"\byou can think about (?:a )?(?:long|short)\b",
    r"\byou could be (?:long|short)\b",
    r"\byou might be able to put (?:a\s+)?(?:piece|peace) on\b",
    r"\byou re looking for a (?:long|short)\b",
    r"\blook to position (?:long|short)\b",
    r"\bi was (?:buying|selling) here\b",
    r"\b(?:but )?i was (?:long|short) from\b",
    r"\b(?:like )?you saw me get (?:long|short) here\b",
    r"\bignoring the (?:long|short)\b",
    r"\bignored the (?:long|short) here\b",
    r"\battempted the (?:long|short) here\b.*\bbreak ?even\b",
    r"\b(?:would not|wouldn t)\s+want to be longing this\b",
    r"\b(?:tempted|so tempted)\s+to (?:try another |just hit |get )?(?:long|short)\b",
    r"\bif you(?: re| are)?\s+(?:buying|selling) here\b",
    r"\b(?:anyone|everyone|who)\s+(?:who\s+)?got (?:long|short) here\b",
    r"\bgot (?:long|short) here has become\b",
    r"\bthink about who s playing (?:long|short) here\b",
    r"\bif you re just joining\b",
    r"\bi wouldn t be adding (?:long|short) here\b",
    r"\bi definitely wouldn t be (?:long|short) here\b",
    r"\brunner break ?even\b.*\bi was looking for\b",
    r"\bi was (?:long|short)\b.*\bthen i took the (?:long|short)\b",
    r"\bwas (?:long|short)\b.*\bthen i took the (?:long|short)\b",
    r"\blet s say you(?: re| are)?\s+(?:buying|selling|long|short)\b",
]
AUTO_ONLY_NON_ACTIONABLE_TRADE_PATTERNS = [
    r"\byou can (?:go|get|be) (?:long|short)\b",
    r"\byou can buy here\b",
    r"\byou can sell here\b",
    r"\byou can pay yourself\b",
    r"\byou can take (?:some )?profit\b",
    r"\blooking for (?:the |a )?(?:long|short)\b",
    r"\binterested in (?:the |a )?(?:long|short)\b",
    r"\btempted to put (?:a\s+)?(?:little|small)?\s*(?:piece|peace)\s+on\b",
    r"\blooking to see if i can get (?:a\s+)?(?:piece|peace)\s+on\b",
    r"\bif you(?: just| got| were| re| are| play)?\s+(?:long|short)\b",
    r"\bif you(?: just| got| were| re| are| play)?\s+.*\b(?:long|short)\b",
    r"\bif you got (?:long|short)\b",
    r"\bif you were (?:long|short)\b",
    r"\bif you play (?:long|short)\b",
    r"\bif you are longing this\b",
    r"\bif you re longing this\b",
    r"\blet s say you re (?:long|short)\b",
    r"\btechnically if you re (?:long|short)\b",
    r"\b(?:they|shorts|buyers|sellers)\s+(?:are|re)\s+piling in (?:long|short)\b",
    r"\byou could try a (?:cheap )?(?:long|short) here\b",
    r"\bfor example\b.*\b(?:long|short) here\b",
]
RETROSPECTIVE_RECOVERY_NEGATIVE_PATTERNS = [
    r"\bi thought\b.*\b(?:put|got)\s+(?:a\s+)?(?:small|little)?\s*(?:piece|peace)\s+on\b",
    r"\b(?:put|got)\s+(?:a\s+)?(?:small|little)?\s*(?:piece|peace)\s+on\b.*\b(?:took|take)\s+a\s+loss\b",
    r"\bdidn t stick to my rules\b",
    r"\bremember\b.*\bi was (?:long|short)\b",
    r"\bi was (?:long|short)\s+(?:on|from)\b",
    r"\bwe(?: re| are)\s+(?:long|short)\s+from\b",
    r"\bi got knocked out\b",
    r"\bi just got knocked out\b",
]
TRIM_PATTERNS = [
    r"\bpaying myself\b",
    r"\bpaid myself\b",
    r"\bpay myself\b",
    r"\btrimming\b",
    r"\bpeeling\b",
    r"\btaking (?:some|a little) off\b",
]
IMPERATIVE_TRIM_PATTERNS = [
    r"\bpay yourself(?: some)?\b",
    r"\btake (?:some )?profit\b",
    r"\btake (?:some|a little) off\b",
    r"\btake (?:a )?little profit\b",
]
IMPERATIVE_TRIM_NEGATIVE_PATTERNS = [
    r"\byou can (?:pay yourself|take (?:some )?profit|take (?:some|a little) off|take (?:a )?little profit)\b",
]
ADVISORY_TRIM_PATTERNS = [
    r"\byou can (?:pay yourself|take (?:some )?profit|take (?:a )?little (?:off|profit))\b",
    r"\btake (?:some )?profit\b",
    r"\bpay yourself\b",
]
ADVISORY_LONG_PATTERNS = [
    r"\byou can (?:go|get) long\b",
    r"\byou can buy here\b",
    r"\byou can (?:get|be) long\b",
]
ADVISORY_SHORT_PATTERNS = [
    r"\byou can (?:go|get) short\b",
    r"\byou can sell here\b",
    r"\byou can (?:get|be) short\b",
]
RUNNER_PATTERNS = [r"\brunners on deck\b", r"\brunner on deck\b", r"\bleave (?:a )?runner\b"]
BREAKEVEN_PATTERNS = [
    r"\bbreak ?even\b",
    r"\brunner break ?even\b",
    r"\bbreak ?even on (?:the )?runner\b",
    r"\bmove stop to be\b",
    r"\bstop to be\b",
    r"\bmoving my stop into the money\b",
    r"\bmove my stop into the money\b",
]
MOVE_STOP_PATTERNS = [
    r"\bmove stop\b",
    r"\bmove my stop\b",
    r"\bmoved my stop\b",
    r"\bstart moving my stop\b",
    r"\bmy stop(?: s| is)?\b",
    r"\bstops? moved to\b",
    r"\bstop (?:is|at|to|under|above|over|below)\b",
    r"\bstay heavy\b",
    r"\bif (?:we|they|this)\s+hold\b",
]
CUT_POSITION_STOP_PATTERNS = [
    r"\b(?:i ll|ill|i will)\s+cut (?:the )?whole position (?:above|below|under|over|at)\s+(?P<price>[\w\.\s]+)",
]
ADD_NOW_PATTERNS = [r"\bgot my add on\b", r"\bgot my ad on\b", r"\bpopped one back on\b", r"\badding here\b", r"\badd on there\b", r"\badd here\b"]
ADD_TENTATIVE_PATTERNS = [
    r"\bconsider starting to add\b",
    r"\blook for an add\b",
    r"\badd on pops\b",
]
RISK_PATTERNS = [
    r"\b(?:long|short)\s+versus\s+(?P<price>[\w\.\s]+)",
    r"\bversus\s+(?P<price>[\w\.\s]+)",
    r"\bstay heavy(?: now)?(?: at| versus)?\s+(?P<price>[\w\.\s]+)",
    r"\bno reclaim(?: of)?\s+(?P<price>[\w\.\s]+)",
    r"\b(?:do not|don t|dont)\s+(?:want to see )?(?:a )?reclaim(?: of)?\s+(?P<price>[\w\.\s]+)",
    r"\bi ll be out on a reclaim(?: of)?\s+(?P<price>[\w\.\s]+)",
    r"\bif (?:we|they)\s+hold\s+(?P<price>[\w\.\s]+)\s+here\b.*\bi m out\b",
    r"\bhold\s+(?P<price>[\w\.\s]+)\s+here\b.*\bi m out\b",
]
STOP_PATTERNS = [
    r"\bstop (?:is |at |to |under |above |over |below )(?P<price>[\w\.\s]+)",
    r"\bmy stop(?: s| is)?(?: gone in)?(?: basically| around| at| to)?\s+(?P<price>[\w\.\s]+)",
    r"\bstops? moved to\s+(?P<price>[\w\.\s]+)",
    r"\bmove(?:d|ing)? my stop(?: up| down| tighter| nice and tight)?(?: to| at)?\s+(?P<price>[\w\.\s]+)",
    r"\brisking (?P<price>[\w\.\s]+)",
]
TARGET_PATTERNS = [
    r"\btarget (?:is |at )(?P<price>[\w\.\s]+)",
    r"\btrim at (?P<price>[\w\.\s]+)",
    r"\bup to (?P<price>[\w\.\s]+)",
    r"\bdown to (?P<price>[\w\.\s]+)",
    r"\blooking for (?P<price>\d{2,5}(?:\.\d{1,2})?s?)\b",
]
ENTRY_PATTERNS = [r"\bat (?P<price>[\w\.\s]+)", r"\bfrom (?P<price>[\w\.\s]+)", r"\bthrough (?P<price>[\w\.\s]+)"]

PRICE_TOKEN_PATTERN = re.compile(r"(?P<num>\d{1,5}(?:\.\d{1,2})?)")
_NUMBER_WORDS: dict[str, int] = {
    "zero": 0,
    "oh": 0,
    "o": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALE_WORDS: dict[str, int] = {"hundred": 100, "thousand": 1000}
_NUMBER_FILLER_WORDS = {"and", "the", "a", "an"}
_GROUP_SEPARATOR_WORDS = {"oh", "o"}
TRADE_KEYWORDS = (
    "long",
    "short",
    "piece",
    "peace",
    "feeler",
    "versus",
    "reclaim",
    "add",
    "ad on",
    "paying myself",
    "paid myself",
    "pay yourself",
    "take profit",
    "profit",
    "yourself",
    "trim",
    "peel",
    "runner",
    "flatten",
    "flat",
    "cut that",
    "stopped out",
    "breakeven",
    "break even",
    "buy",
    "sell",
)
NON_ACTIONABLE_ENTRY_PATTERNS = [
    r"\bwas it this (?:long|short) here\b",
    r"\bhow am i going to get (?:long|short)\b",
    r"\bfind a way to get (?:long|short)\b",
    r"\b(?:we|you|i)\s+can\s+look\s+for\s+(?:this\s+)?(?:long|short)\b",
    r"\bcan\s+look\s+for\s+(?:this\s+)?(?:long|short)\b",
    r"\bthink about taking a (?:long|short)\b",
    r"\b(?:do not|don t|dont)\s+want to (?:get |be )?(?:long|short)\b",
    r"\bdidn t want to (?:get |be )?(?:long|short)\b",
    r"\bwouldn t want to (?:get |be )?(?:long|short)\b",
    r"\b(?:do not|don t|dont)\s+want to be (?:entering|get(?:ting)?|adding|bet(?:ting)?)\s+(?:long|short)\b",
    r"\b(?:do not|don t|dont)\s+want to just (?:long|short)\b",
    r"\b(?:we ve|we have)\s+had\s+a\s+(?:huge|big)\s+(?:long|short)\b",
    r"\bi was (?:long|short) here\b",
    r"\byou re adding (?:long|short) here\b",
    r"\bi m long on my swings\b",
    r"\bi m long in all the software companies\b",
    r"\bi m literally long in all of those\b",
]
FOREIGN_CONTEXT_PATTERNS = [
    r"\bmulti year\b",
    r"\bmy swings?\b",
    r"\bsoftware companies\b",
    r"\ball the software\b",
    r"\bstocks that i m long in\b",
    r"\bstill holding this long on\b",
]
SESSION_SYMBOL_ALIASES: dict[str, set[str]] = {
    "NQ": {"nq", "mnq"},
    "MNQ": {"nq", "mnq"},
    "ES": {"es", "mes"},
    "MES": {"es", "mes"},
    "YM": {"ym", "mym"},
    "MYM": {"ym", "mym"},
    "RTY": {"rty", "m2k"},
    "M2K": {"rty", "m2k"},
    "CL": {"cl", "mcl", "oil", "crude"},
    "MCL": {"cl", "mcl", "oil", "crude"},
    "GC": {"gc", "mgc", "gold"},
    "MGC": {"gc", "mgc", "gold"},
    "SI": {"si", "silver"},
    "SIL": {"si", "silver"},
}
FOREIGN_INSTRUMENT_TOKENS = {
    "aapl",
    "amd",
    "amzn",
    "bitcoin",
    "btc",
    "crude",
    "eth",
    "ethereum",
    "gold",
    "hood",
    "meta",
    "msft",
    "nflx",
    "netflix",
    "nvda",
    "oil",
    "orcl",
    "oracle",
    "paypal",
    "silver",
    "tsla",
    "unh",
}


@dataclass
class ParseContext:
    text: str
    normalized: str
    market_price: float | None
    received_at: datetime
    source_latency_ms: int
    segment_id: str


@dataclass
class PendingEntry:
    side: TradeSide
    created_at: datetime
    seed_price: float | None
    entry_price: float | None
    evidence_text: str


@dataclass
class RecentFragment:
    text: str
    normalized: str
    received_at: datetime
    segment_id: str


@dataclass
class CandidateWindow:
    opened_at: datetime
    updated_at: datetime
    probability: float
    source: str
    side_hint: TradeSide | None = None
    tag_hint: ActionTag | None = None
    fragments: list[RecentFragment] = field(default_factory=list)


@dataclass
class FlowState:
    pending_entry: PendingEntry | None = None
    last_management_at: datetime | None = None
    last_exit_at: datetime | None = None
    last_side: TradeSide | None = None
    active_instrument: str | None = None
    active_instrument_at: datetime | None = None
    recent_text: str | None = None
    recent_text_at: datetime | None = None
    recent_fragments: list[RecentFragment] = field(default_factory=list)
    candidate_window: CandidateWindow | None = None
    recent_intent_tag: ActionTag | None = None
    recent_intent_side: TradeSide | None = None
    recent_intent_at: datetime | None = None
    recent_intent_entry_price: float | None = None
    recent_intent_stop_price: float | None = None
    recent_intent_target_price: float | None = None


@dataclass
class InterpreterDiagnostic:
    event_type: EventType
    title: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfirmationDecision:
    approved: bool
    reason: str | None = None
    fail_open_applied: bool = False


class RuleBasedTradeInterpreter:
    def __init__(
        self,
        *,
        fallback: GeminiFallbackInterpreter | None = None,
        embedding_gate: EmbeddingGate | None = None,
        local_classifier: ModernBertIntentClassifier | None = None,
        classifier_min_probability: float = 0.62,
        classifier_block_probability: float = 0.84,
        classifier_recovery_probability: float = 0.9,
        candidate_window_ms: int = 6_000,
        candidate_preroll_ms: int = 2_400,
        candidate_max_fragments: int = 5,
        candidate_open_probability: float = 0.43,
        candidate_keep_probability: float = 0.28,
        entry_context_window_ms: int = 15_000,
        entry_guard_window_ms: int = 20_000,
        instrument_context_window_ms: int = 30_000,
        context_stitch_window_ms: int = 4_000,
        duplicate_intent_window_ms: int = 2_500,
        duplicate_entry_window_ms: int = 8_000,
        fallback_confirmation_timeout_ms: int = 1_500,
    ) -> None:
        self._fallback = fallback
        self._embedding_gate = embedding_gate
        self._local_classifier = local_classifier
        self._classifier_min_probability = max(0.0, min(1.0, classifier_min_probability))
        self._classifier_block_probability = max(0.0, min(1.0, classifier_block_probability))
        self._classifier_recovery_probability = max(0.0, min(1.0, classifier_recovery_probability))
        self._candidate_window = timedelta(milliseconds=max(1_000, candidate_window_ms))
        self._candidate_preroll_window = timedelta(milliseconds=max(0, min(candidate_preroll_ms, candidate_window_ms)))
        self._candidate_max_fragments = max(2, candidate_max_fragments)
        self._candidate_open_probability = max(0.0, min(1.0, candidate_open_probability))
        self._candidate_keep_probability = max(0.0, min(1.0, candidate_keep_probability))
        self._entry_context_window = timedelta(milliseconds=max(1, entry_context_window_ms))
        self._entry_guard_window = timedelta(milliseconds=max(1, entry_guard_window_ms))
        self._instrument_context_window = timedelta(milliseconds=max(1, instrument_context_window_ms))
        self._context_stitch_window = timedelta(milliseconds=max(1, context_stitch_window_ms))
        self._duplicate_intent_window = timedelta(milliseconds=max(1, duplicate_intent_window_ms))
        self._duplicate_entry_window = timedelta(milliseconds=max(1, duplicate_entry_window_ms))
        self._fallback_confirmation_timeout_s = max(0.1, fallback_confirmation_timeout_ms / 1000.0)
        self._states: dict[str, FlowState] = {}
        self._diagnostics: dict[str, InterpreterDiagnostic] = {}

    def clear_session(self, session_id: str) -> None:
        self._states.pop(session_id, None)
        self._diagnostics.pop(session_id, None)

    def consume_diagnostic(self, session_id: str) -> InterpreterDiagnostic | None:
        return self._diagnostics.pop(session_id, None)

    async def close(self) -> None:
        if self._fallback is not None and hasattr(self._fallback, "close"):
            await self._fallback.close()
        if self._local_classifier is not None and hasattr(self._local_classifier, "close"):
            self._local_classifier.close()

    async def interpret(self, session: StreamSession, segment: TranscriptSegment) -> TradeIntent | None:
        compacted_text = _compact_repeated_trade_text(segment.text)
        self._diagnostics.pop(session.id, None)
        context = ParseContext(
            text=compacted_text,
            normalized=_normalize(compacted_text),
            market_price=session.market.last_price,
            received_at=_coerce_received_at(segment),
            source_latency_ms=segment.metrics.total_latency_ms if segment.metrics else 0,
            segment_id=segment.id,
        )

        state_before = self._get_state(session.id, mutate_state=False)
        analysis_text = self._analysis_text(state_before, text=context.normalized, received_at=context.received_at)
        entry_text = self._entry_text(state_before, text=context.normalized, received_at=context.received_at)
        classifier_prediction = self._classify_local_intent(
            session=session,
            context=context,
            state=state_before,
            analysis_text=analysis_text,
            entry_text=entry_text,
        )
        intent = self._parse(session=session, context=context, mutate_state=True)
        intent = self._apply_local_classifier(
            session=session,
            context=context,
            state_before=state_before,
            analysis_text=analysis_text,
            entry_text=entry_text,
            parser_intent=intent,
            classifier_prediction=classifier_prediction,
        )
        state_after_parse = self._get_state(session.id, mutate_state=True)
        candidate_assessment = self._assess_candidate(
            session=session,
            context=context,
            analysis_text=analysis_text,
            entry_text=entry_text,
            parser_intent=intent,
            classifier_prediction=classifier_prediction,
        )
        self._update_candidate_tracking(
            state=state_after_parse,
            context=context,
            candidate_assessment=candidate_assessment,
            intent=intent,
        )
        if intent is None or intent.tag in {ActionTag.setup_long, ActionTag.setup_short}:
            recovered_intent = self._recover_from_candidate_window(
                session=session,
                context=context,
                parser_intent=intent,
                classifier_prediction=classifier_prediction,
            )
            if recovered_intent is not None:
                intent = recovered_intent
        if intent is not None and self._should_confirm_with_fallback(session, intent):
            confirmation = await self._confirm_with_fallback(
                session=session,
                segment=segment,
                context=context,
                intent=intent,
                state_before=state_before,
            )
            if not confirmation.approved:
                self._diagnostics[session.id] = InterpreterDiagnostic(
                    event_type=EventType.warning,
                    title="Entry confirmation blocked",
                    message=confirmation.reason or "Gemini did not confirm the entry.",
                    data={
                        "candidate_intent": intent.model_dump(mode="json"),
                        "compacted_text": context.text,
                    },
                )
                return None
            if confirmation.fail_open_applied:
                self._diagnostics[session.id] = InterpreterDiagnostic(
                    event_type=EventType.system,
                    title="Entry confirmation degraded",
                    message=confirmation.reason or "Gemini confirmation degraded; strong rule entry allowed.",
                    data={"candidate_intent": intent.model_dump(mode="json"), "compacted_text": context.text},
                )
        if intent is not None and not self._should_clarify(context.normalized, intent):
            return intent
        if not session.config.enable_ai_fallback or self._fallback is None:
            return intent
        if not self._should_use_extractive_fallback(session):
            return intent
        trade_relevant = self._looks_trade_candidate(context.normalized)
        if self._embedding_gate is not None:
            try:
                trade_relevant = self._embedding_gate.is_trade_relevant(context.normalized)
            except Exception:
                # Fail-open to keyword gate if the embedding backend is unavailable.
                trade_relevant = self._looks_trade_candidate(context.normalized)
        if not trade_relevant:
            return intent
        clarified = await self._fallback.interpret(session=session, segment=segment)
        return clarified or intent

    def interpret_partial(self, session: StreamSession, segment: TranscriptSegment) -> TradeIntent | None:
        compacted_text = _compact_repeated_trade_text(segment.text)
        context = ParseContext(
            text=compacted_text,
            normalized=_normalize(compacted_text),
            market_price=session.market.last_price,
            received_at=_coerce_received_at(segment),
            source_latency_ms=segment.metrics.total_latency_ms if segment.metrics else 0,
            segment_id=segment.id,
        )
        if not self._looks_trade_candidate(context.normalized):
            return None

        intent = self._parse(session=session, context=context, mutate_state=False)
        if intent is None:
            return None
        if intent.tag not in {
            ActionTag.enter_long,
            ActionTag.enter_short,
            ActionTag.add,
            ActionTag.trim,
            ActionTag.exit_all,
            ActionTag.move_stop,
            ActionTag.move_to_breakeven,
        }:
            return None
        if intent.confidence < 0.74:
            return None
        return intent

    def interpret_preview_entry(self, session: StreamSession, segment: TranscriptSegment) -> TradeIntent | None:
        if session.position is not None:
            return None
        compacted_text = _compact_repeated_trade_text(segment.text)
        context = ParseContext(
            text=compacted_text,
            normalized=_normalize(compacted_text),
            market_price=session.market.last_price,
            received_at=_coerce_received_at(segment),
            source_latency_ms=segment.metrics.total_latency_ms if segment.metrics else 0,
            segment_id=segment.id,
        )
        if not self._looks_trade_candidate(context.normalized):
            return None
        intent = self._parse(session=session, context=context, mutate_state=False)
        if intent is None or intent.tag not in {ActionTag.enter_long, ActionTag.enter_short}:
            return None
        if intent.guard_reason is not None or intent.stop_price is None or intent.confidence < 0.9:
            return None
        if not self._is_strong_preview_entry_text(context.normalized, side=intent.side):
            return None
        return intent

    def confirm_preview_entry(
        self,
        session: StreamSession,
        segment: TranscriptSegment,
        *,
        pending_intent: TradeIntent,
    ) -> bool:
        if pending_intent.side is None or pending_intent.stop_price is None:
            return False
        compacted_text = _compact_repeated_trade_text(segment.text)
        context = ParseContext(
            text=compacted_text,
            normalized=_normalize(compacted_text),
            market_price=session.market.last_price,
            received_at=_coerce_received_at(segment),
            source_latency_ms=segment.metrics.total_latency_ms if segment.metrics else 0,
            segment_id=segment.id,
        )
        if self._is_non_actionable_entry_commentary(context.normalized) or self._is_non_actionable_trade_commentary(
            context.normalized,
            session=session,
        ):
            return False
        if not self._is_strong_preview_entry_text(context.normalized, side=pending_intent.side):
            return False
        stop_price = self._find_price(context.normalized, context, RISK_PATTERNS + STOP_PATTERNS)
        if stop_price is None:
            return False
        return _price_equal(stop_price, pending_intent.stop_price)

    def _parse(self, *, session: StreamSession, context: ParseContext, mutate_state: bool) -> TradeIntent | None:
        text = context.normalized
        state = self._get_state(session.id, mutate_state=mutate_state)
        self._prune_temporal_state(state, received_at=context.received_at, mutate_state=mutate_state)
        analysis_text = self._analysis_text(state, text=text, received_at=context.received_at)
        entry_text = self._entry_text(state, text=text, received_at=context.received_at)
        if mutate_state:
            state.recent_text = text
            state.recent_text_at = context.received_at
            self._remember_recent_fragment(state, context=context)
        return self._parse_compiled(
            session=session,
            context=context,
            state=state,
            analysis_text=analysis_text,
            entry_text=entry_text,
            mutate_state=mutate_state,
        )

    def _parse_compiled(
        self,
        *,
        session: StreamSession,
        context: ParseContext,
        state: FlowState,
        analysis_text: str,
        entry_text: str,
        mutate_state: bool,
    ) -> TradeIntent | None:
        text = context.normalized
        if session.position is not None:
            state.last_side = session.position.side
        pending = self._get_pending_entry(state, received_at=context.received_at, mutate_state=mutate_state)
        if self._mentions_session_instrument(analysis_text, session):
            state.active_instrument = None
            state.active_instrument_at = None
        else:
            foreign_instrument = self._find_foreign_instrument(analysis_text, session)
            if foreign_instrument is not None:
                state.active_instrument = foreign_instrument
                state.active_instrument_at = context.received_at
        if self._active_foreign_instrument(state, received_at=context.received_at) is not None and self._looks_trade_candidate(analysis_text):
            return None
        if self._is_non_actionable_trade_commentary(analysis_text, session=session) or self._is_non_actionable_entry_commentary(analysis_text):
            if not self._allows_setup_seed_entry_override(session=session, state=state, context=context, text=text):
                return None
        side_from_text = self._detect_side(text)
        if side_from_text is not None:
            state.last_side = side_from_text

        if "cancel that" in text or "scratch that" in text:
            if mutate_state:
                state.pending_entry = None
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.cancel_setup,
                side=side_from_text,
                confidence=0.8,
                state=state,
                mutate_state=mutate_state,
            )

        if _matches_any(NON_ACTIONABLE_EXIT_PATTERNS, analysis_text):
            return None

        conditional_exit_stop = self._find_price(analysis_text, context, RISK_PATTERNS + STOP_PATTERNS)
        if _matches_any(CONDITIONAL_EXIT_PATTERNS, analysis_text):
            if conditional_exit_stop is not None and session.position is not None and (
                "versus" in analysis_text
                or "reclaim" in analysis_text
                or "stay heavy" in analysis_text
                or _matches_any(MOVE_STOP_PATTERNS, analysis_text)
            ):
                if mutate_state:
                    state.pending_entry = None
                    state.last_management_at = context.received_at
                return self._build_intent(
                    session=session,
                    context=context,
                    tag=ActionTag.move_stop,
                    side=session.position.side,
                    stop_price=conditional_exit_stop,
                    state=state,
                    mutate_state=mutate_state,
                )
            return None

        if _matches_any(EXIT_PATTERNS, entry_text):
            if mutate_state:
                state.pending_entry = None
                state.last_exit_at = context.received_at
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.exit_all,
                side=session.position.side if session.position is not None else side_from_text or state.last_side,
                quantity_hint="all",
                state=state,
                mutate_state=mutate_state,
            )

        cut_stop = self._find_price(entry_text, context, CUT_POSITION_STOP_PATTERNS)
        if cut_stop is not None:
            if mutate_state:
                state.pending_entry = None
                state.last_management_at = context.received_at
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.move_stop,
                side=session.position.side if session.position is not None else side_from_text or state.last_side,
                stop_price=cut_stop,
                state=state,
                mutate_state=mutate_state,
            )

        if _matches_any(BREAKEVEN_PATTERNS, entry_text):
            if mutate_state:
                state.pending_entry = None
                state.last_management_at = context.received_at
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.move_to_breakeven,
                side=session.position.side if session.position is not None else side_from_text or state.last_side,
                state=state,
                mutate_state=mutate_state,
            )

        risk_stop = self._find_price(text, context, RISK_PATTERNS + STOP_PATTERNS)
        target_price = self._find_price(text, context, TARGET_PATTERNS)
        entry_price = self._find_price(text, context, ENTRY_PATTERNS)
        entry_risk_stop = self._find_price(entry_text, context, RISK_PATTERNS + STOP_PATTERNS)
        entry_target_price = self._find_price(entry_text, context, TARGET_PATTERNS)
        stitched_entry_price = self._find_price(entry_text, context, ENTRY_PATTERNS)
        resolved_trim_target = target_price if target_price is not None else entry_price if entry_price is not None else context.market_price

        if session.position is not None and self._is_actionable_trim_command(entry_text):
            if mutate_state:
                state.pending_entry = None
                state.last_management_at = context.received_at
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.trim,
                side=session.position.side,
                target_price=resolved_trim_target,
                quantity_hint="some",
                state=state,
                mutate_state=mutate_state,
            )

        if _matches_any(ADVISORY_TRIM_PATTERNS, analysis_text) and self._advisory_intents_enabled(session):
            if mutate_state:
                state.pending_entry = None
                state.last_management_at = context.received_at
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.trim,
                side=session.position.side if session.position is not None else side_from_text or state.last_side,
                target_price=resolved_trim_target,
                quantity_hint="some",
                state=state,
                mutate_state=mutate_state,
            )

        advisory_long = _matches_any(ADVISORY_LONG_PATTERNS, analysis_text) and self._advisory_intents_enabled(session)
        advisory_short = _matches_any(ADVISORY_SHORT_PATTERNS, analysis_text) and self._advisory_intents_enabled(session)
        if advisory_long or advisory_short:
            adv_side = TradeSide.long if advisory_long else TradeSide.short
            if mutate_state:
                state.pending_entry = None
                state.last_side = adv_side
            return self._build_intent(
                session=session,
                context=context,
                tag=self._entry_tag_for_side(adv_side, session),
                side=adv_side,
                entry_price=entry_price if entry_price is not None else context.market_price,
                stop_price=risk_stop,
                target_price=target_price,
                quantity_hint=self._detect_entry_size_hint(text),
                guard_reason=self._entry_guard_reason(state, context.received_at),
                state=state,
                mutate_state=mutate_state,
            )

        if _matches_any(TRIM_PATTERNS, entry_text) or _matches_any(RUNNER_PATTERNS, entry_text):
            if mutate_state:
                state.pending_entry = None
                state.last_management_at = context.received_at
            quantity_hint = "most" if "most my size" in entry_text or _matches_any(RUNNER_PATTERNS, entry_text) else self._detect_quantity(entry_text)
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.trim,
                side=session.position.side if session.position is not None else side_from_text or state.last_side,
                target_price=resolved_trim_target,
                quantity_hint=quantity_hint,
                state=state,
                mutate_state=mutate_state,
            )

        add_now = _matches_any(ADD_NOW_PATTERNS, entry_text) and not _matches_any(ADD_TENTATIVE_PATTERNS, entry_text)
        if add_now and session.position is not None:
            if mutate_state:
                state.pending_entry = None
                state.last_side = session.position.side
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.add,
                side=session.position.side,
                entry_price=stitched_entry_price if stitched_entry_price is not None else context.market_price,
                stop_price=entry_risk_stop if entry_risk_stop is not None else session.position.stop_price,
                quantity_hint=self._detect_quantity(entry_text),
                state=state,
                mutate_state=mutate_state,
            )

        entry_side = self._detect_entry_side(
            entry_text,
            market_price=context.market_price,
            target_price=entry_target_price,
        )
        entry_trigger = self._is_entry_trigger(entry_text) or (add_now and session.position is None)
        if entry_trigger:
            side = entry_side or side_from_text or state.last_side
            if side is None:
                side = self._active_setup_side_hint(state, received_at=context.received_at)
            if side is None and pending is not None:
                side = pending.side
            if side is not None:
                if mutate_state:
                    state.pending_entry = None
                    state.last_side = side
                return self._build_intent(
                    session=session,
                    context=context,
                    tag=self._entry_tag_for_side(side, session),
                    side=side,
                    entry_price=stitched_entry_price if stitched_entry_price is not None else context.market_price,
                    stop_price=entry_risk_stop,
                    target_price=entry_target_price,
                    quantity_hint=self._detect_entry_size_hint(entry_text),
                    guard_reason=self._entry_guard_reason(state, context.received_at),
                    state=state,
                    mutate_state=mutate_state,
                )

            return None

        if pending is not None and risk_stop is not None:
            if mutate_state:
                state.pending_entry = None
                state.last_side = pending.side
            stitched_evidence = f"{pending.evidence_text} | {context.text.strip()}" if pending.evidence_text else context.text.strip()
            return self._build_intent(
                session=session,
                context=context,
                tag=self._entry_tag_for_side(pending.side, session),
                side=pending.side,
                entry_price=stitched_entry_price if stitched_entry_price is not None else pending.entry_price or pending.seed_price or context.market_price,
                stop_price=entry_risk_stop,
                target_price=entry_target_price,
                quantity_hint=self._detect_entry_size_hint(entry_text) or "one",
                guard_reason=self._entry_guard_reason(state, context.received_at),
                evidence_text=stitched_evidence,
                state=state,
                mutate_state=mutate_state,
            )

        if risk_stop is not None and session.position is not None and (
            "versus" in entry_text or "reclaim" in entry_text or _matches_any(MOVE_STOP_PATTERNS, entry_text)
        ):
            if mutate_state:
                state.last_management_at = context.received_at
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.move_stop,
                side=session.position.side,
                stop_price=risk_stop,
                state=state,
                mutate_state=mutate_state,
            )

        if _matches_any(SETUP_LONG_PATTERNS, text):
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.setup_long,
                side=TradeSide.long,
                confidence=0.72,
                state=state,
                mutate_state=mutate_state,
            )
        if _matches_any(SETUP_SHORT_PATTERNS, text):
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.setup_short,
                side=TradeSide.short,
                confidence=0.72,
                state=state,
                mutate_state=mutate_state,
            )

        return None

    def _build_intent(
        self,
        *,
        session: StreamSession,
        context: ParseContext,
        tag: ActionTag,
        side: TradeSide | None,
        entry_price: float | None = None,
        stop_price: float | None = None,
        target_price: float | None = None,
        quantity_hint: str | None = None,
        guard_reason: str | None = None,
        confidence: float | None = None,
        evidence_text: str | None = None,
        state: FlowState | None = None,
        mutate_state: bool = False,
    ) -> TradeIntent | None:
        score = confidence if confidence is not None else self._score(
            tag=tag,
            text=context.normalized,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            guard_reason=guard_reason,
        )
        intent = TradeIntent(
            session_id=session.id,
            tag=tag,
            symbol=session.market.symbol,
            side=side,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            quantity_hint=quantity_hint,
            confidence=score,
            evidence_text=(evidence_text or context.text).strip(),
            source_segment_id=context.segment_id,
            source_received_at=context.received_at,
            source_latency_ms=max(0, context.source_latency_ms),
            guard_reason=guard_reason,
        )
        if state is not None and self._should_suppress_duplicate_intent(state, context=context, intent=intent):
            return None
        if state is not None and mutate_state:
            self._remember_intent(state, context=context, intent=intent)
        return intent

    def _classify_local_intent(
        self,
        *,
        session: StreamSession,
        context: ParseContext,
        state: FlowState,
        analysis_text: str,
        entry_text: str,
    ) -> IntentClassifierPrediction | None:
        if self._local_classifier is None or not self._local_classifier.is_available():
            return None
        if not (
            self._looks_trade_candidate(analysis_text)
            or looks_candidate_seed(analysis_text)
            or looks_candidate_seed(entry_text)
            or detect_present_trade_signal(entry_text, position_side=session.position.side if session.position is not None else None)
            is not None
            or detect_setup_signal(analysis_text) is not None
            or self._has_active_candidate_window(state, received_at=context.received_at)
        ):
            return None
        envelope = IntentContextEnvelope(
            symbol=session.market.symbol,
            current_text=context.text,
            current_normalized=context.normalized,
            recent_text=state.recent_text,
            analysis_text=analysis_text,
            entry_text=entry_text,
            position_side=session.position.side if session.position is not None else None,
            last_side=state.last_side,
            market_price=context.market_price,
        )
        return self._local_classifier.classify(envelope)

    def _classify_window_intent(
        self,
        *,
        session: StreamSession,
        context: ParseContext,
        state: FlowState,
    ) -> IntentClassifierPrediction | None:
        if self._local_classifier is None or not self._local_classifier.is_available():
            return None
        envelope = IntentContextEnvelope(
            symbol=session.market.symbol,
            current_text=context.text,
            current_normalized=context.normalized,
            recent_text=None,
            analysis_text=context.normalized,
            entry_text=context.normalized,
            position_side=session.position.side if session.position is not None else None,
            last_side=state.last_side,
            market_price=context.market_price,
        )
        return self._local_classifier.classify(envelope)

    def _assess_candidate(
        self,
        *,
        session: StreamSession,
        context: ParseContext,
        analysis_text: str,
        entry_text: str,
        parser_intent: TradeIntent | None,
        classifier_prediction: IntentClassifierPrediction | None,
    ) -> CandidateAssessment | None:
        position_side = session.position.side if session.position is not None else None
        explicit_signal = detect_present_trade_signal(entry_text, position_side=position_side) or detect_present_trade_signal(
            analysis_text,
            position_side=position_side,
        )
        setup_signal = detect_setup_signal(analysis_text)
        assessment = assess_trade_candidate(
            text=context.normalized,
            analysis_text=analysis_text,
            entry_text=entry_text,
            classifier_prediction=classifier_prediction,
            explicit_signal=explicit_signal,
            setup_signal=setup_signal,
        )
        if parser_intent is not None and parser_intent.tag in {ActionTag.setup_long, ActionTag.setup_short}:
            parser_assessment = CandidateAssessment(
                probability=0.74,
                source="parser_setup",
                tag_hint=parser_intent.tag,
                side_hint=parser_intent.side,
            )
            if assessment is None or parser_assessment.probability >= assessment.probability:
                return parser_assessment
        return assessment

    def _update_candidate_tracking(
        self,
        *,
        state: FlowState,
        context: ParseContext,
        candidate_assessment: CandidateAssessment | None,
        intent: TradeIntent | None,
    ) -> None:
        self._prune_temporal_state(state, received_at=context.received_at, mutate_state=True)
        if intent is not None and intent.tag not in {ActionTag.setup_long, ActionTag.setup_short}:
            state.candidate_window = None
            return

        window_active = self._has_active_candidate_window(state, received_at=context.received_at)
        if candidate_assessment is None:
            if not window_active or not looks_candidate_continuation(context.normalized):
                return
            candidate_assessment = CandidateAssessment(
                probability=self._candidate_keep_probability,
                source="continuation",
                side_hint=state.candidate_window.side_hint if state.candidate_window is not None else None,
                tag_hint=state.candidate_window.tag_hint if state.candidate_window is not None else None,
            )
        if not candidate_assessment.should_track(
            open_threshold=self._candidate_open_probability,
            keep_threshold=self._candidate_keep_probability,
            window_active=window_active,
        ):
            return
        self._track_candidate_window(state, context=context, assessment=candidate_assessment)

    def _recover_from_candidate_window(
        self,
        *,
        session: StreamSession,
        context: ParseContext,
        parser_intent: TradeIntent | None,
        classifier_prediction: IntentClassifierPrediction | None,
    ) -> TradeIntent | None:
        state = self._get_state(session.id, mutate_state=True)
        window = self._active_candidate_window(state, received_at=context.received_at, mutate_state=True)
        if window is None or len(window.fragments) < 2:
            return None

        window_context = self._build_candidate_window_context(window=window, context=context)
        if window_context is None or window_context.normalized == context.normalized:
            return None

        state_before = self._get_state(session.id, mutate_state=False)
        window_prediction = self._classify_window_intent(session=session, context=window_context, state=state)
        recovered = self._parse_compiled(
            session=session,
            context=window_context,
            state=state,
            analysis_text=window_context.normalized,
            entry_text=window_context.normalized,
            mutate_state=True,
        )
        recovered = self._apply_local_classifier(
            session=session,
            context=window_context,
            state_before=state_before,
            analysis_text=window_context.normalized,
            entry_text=window_context.normalized,
            parser_intent=recovered,
            classifier_prediction=window_prediction or classifier_prediction,
        )
        if recovered is None or recovered.tag in {ActionTag.setup_long, ActionTag.setup_short, ActionTag.cancel_setup}:
            return None
        if recovered.tag in {ActionTag.enter_long, ActionTag.enter_short, ActionTag.add} and self._is_retrospective_recovery_text(
            window_context.normalized
        ):
            return None

        state.candidate_window = None
        if session.id not in self._diagnostics:
            recovered_tag = recovered.tag.value if isinstance(recovered.tag, ActionTag) else str(recovered.tag)
            self._diagnostics[session.id] = InterpreterDiagnostic(
                event_type=EventType.system,
                title="Candidate window recovered intent",
                message=f"Stitched candidate follow-through recovered a {recovered_tag.lower()} intent.",
                data={
                    "candidate_window": self._candidate_window_payload(window),
                    "recovered_intent": recovered.model_dump(mode="json"),
                    "classifier": (
                        self._classifier_prediction_payload(window_prediction)
                        if window_prediction is not None
                        else None
                    ),
                },
            )
        return recovered

    def _apply_local_classifier(
        self,
        *,
        session: StreamSession,
        context: ParseContext,
        state_before: FlowState,
        analysis_text: str,
        entry_text: str,
        parser_intent: TradeIntent | None,
        classifier_prediction: IntentClassifierPrediction | None,
    ) -> TradeIntent | None:
        if classifier_prediction is None:
            return parser_intent

        intent = parser_intent
        if intent is not None:
            if self._classifier_blocks_intent(intent=intent, prediction=classifier_prediction, text=entry_text):
                intent_tag = intent.tag.value if isinstance(intent.tag, ActionTag) else str(intent.tag)
                self._restore_state_after_veto(session.id, state_before)
                self._diagnostics[session.id] = InterpreterDiagnostic(
                    event_type=EventType.warning,
                    title="Local classifier blocked intent",
                    message=f"{classifier_prediction.model_name} rejected {intent_tag.lower()} on this segment.",
                    data={
                        "candidate_intent": intent.model_dump(mode="json"),
                        "classifier": self._classifier_prediction_payload(classifier_prediction),
                    },
                )
                return None
            intent.confidence = max(intent.confidence, self._classifier_support_probability(intent, classifier_prediction))
            return intent

        guided = self._classifier_guided_intent(
            session=session,
            context=context,
            analysis_text=analysis_text,
            entry_text=entry_text,
            classifier_prediction=classifier_prediction,
        )
        if guided is not None:
            guided_tag = guided.tag.value if isinstance(guided.tag, ActionTag) else str(guided.tag)
            self._diagnostics[session.id] = InterpreterDiagnostic(
                event_type=EventType.system,
                title="Local classifier promoted intent",
                message=f"{classifier_prediction.model_name} recovered a {guided_tag.lower()} intent from explicit self-action language.",
                data={
                    "guided_intent": guided.model_dump(mode="json"),
                    "classifier": self._classifier_prediction_payload(classifier_prediction),
                },
            )
        return guided

    def _classifier_blocks_intent(
        self,
        *,
        intent: TradeIntent,
        prediction: IntentClassifierPrediction,
        text: str,
    ) -> bool:
        support_probability = self._classifier_support_probability(intent, prediction)
        support_threshold = prediction.threshold_for(
            intent.tag,
            fallback=self._classifier_min_probability,
        )
        if support_probability >= support_threshold:
            return False

        strong_rule = intent.confidence >= 0.92
        if intent.tag in {ActionTag.enter_long, ActionTag.enter_short}:
            strong_rule = strong_rule or (
                intent.stop_price is not None and self._is_strong_preview_entry_text(text, side=intent.side)
            )
        if strong_rule:
            return False

        if prediction.non_action_probability >= self._classifier_block_probability:
            return True

        return False

    def _classifier_support_probability(
        self,
        intent: TradeIntent,
        prediction: IntentClassifierPrediction,
    ) -> float:
        direct = prediction.probability_for(intent.tag)
        if intent.tag == ActionTag.enter_long and intent.side == TradeSide.long:
            return max(direct, prediction.probability_for(ActionTag.add))
        if intent.tag == ActionTag.enter_short and intent.side == TradeSide.short:
            return max(direct, prediction.probability_for(ActionTag.add))
        if intent.tag == ActionTag.add:
            if intent.side == TradeSide.long:
                return max(direct, prediction.probability_for(ActionTag.enter_long))
            if intent.side == TradeSide.short:
                return max(direct, prediction.probability_for(ActionTag.enter_short))
        return direct

    def _classifier_guided_intent(
        self,
        *,
        session: StreamSession,
        context: ParseContext,
        analysis_text: str,
        entry_text: str,
        classifier_prediction: IntentClassifierPrediction,
    ) -> TradeIntent | None:
        signal = detect_present_trade_signal(
            analysis_text,
            position_side=session.position.side if session.position is not None else None,
        ) or detect_present_trade_signal(
            entry_text,
            position_side=session.position.side if session.position is not None else None,
        )
        if signal is None or not signal.actionable:
            return None
        if classifier_prediction.confidence < max(
            self._classifier_recovery_probability,
            self._classifier_recovery_floor(signal.tag),
            classifier_prediction.threshold_for(
                classifier_prediction.tag,
                fallback=self._classifier_min_probability,
            ),
        ):
            return None
        if not self._classifier_signal_matches(signal.tag, classifier_prediction.tag):
            return None

        state = self._get_state(session.id, mutate_state=True)
        risk_stop = self._find_price(entry_text, context, RISK_PATTERNS + STOP_PATTERNS)
        target_price = self._find_price(entry_text, context, TARGET_PATTERNS)
        entry_price = self._find_price(entry_text, context, ENTRY_PATTERNS)

        if signal.tag in {ActionTag.enter_long, ActionTag.enter_short, ActionTag.add}:
            side = signal.side or self._side_for_classifier_tag(classifier_prediction.tag)
            if side is None:
                return None
            tag = ActionTag.add if session.position is not None and session.position.side == side else self._entry_tag_for_side(side, session)
            return self._build_intent(
                session=session,
                context=context,
                tag=tag,
                side=side,
                entry_price=entry_price if entry_price is not None else context.market_price,
                stop_price=risk_stop,
                target_price=target_price,
                quantity_hint=self._detect_entry_size_hint(entry_text),
                guard_reason=self._entry_guard_reason(state, context.received_at),
                confidence=max(0.82, classifier_prediction.confidence),
                state=state,
                mutate_state=True,
            )

        if signal.tag == ActionTag.exit_all:
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.exit_all,
                side=session.position.side if session.position is not None else signal.side or state.last_side,
                quantity_hint="all",
                confidence=max(0.86, classifier_prediction.confidence),
                state=state,
                mutate_state=True,
            )

        if signal.tag == ActionTag.trim:
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.trim,
                side=session.position.side if session.position is not None else signal.side or state.last_side,
                target_price=target_price if target_price is not None else context.market_price,
                quantity_hint=self._detect_quantity(entry_text) or "some",
                confidence=max(0.83, classifier_prediction.confidence),
                state=state,
                mutate_state=True,
            )

        if signal.tag == ActionTag.move_to_breakeven:
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.move_to_breakeven,
                side=session.position.side if session.position is not None else signal.side or state.last_side,
                confidence=max(0.84, classifier_prediction.confidence),
                state=state,
                mutate_state=True,
            )

        if signal.tag == ActionTag.move_stop and risk_stop is not None:
            return self._build_intent(
                session=session,
                context=context,
                tag=ActionTag.move_stop,
                side=session.position.side if session.position is not None else signal.side or state.last_side,
                stop_price=risk_stop,
                confidence=max(0.83, classifier_prediction.confidence),
                state=state,
                mutate_state=True,
            )
        return None

    def _classifier_recovery_floor(self, tag: ActionTag) -> float:
        if tag in {ActionTag.enter_long, ActionTag.enter_short, ActionTag.add}:
            return 0.9
        if tag in {ActionTag.exit_all, ActionTag.trim}:
            return 0.88
        if tag == ActionTag.move_stop:
            return 0.86
        if tag == ActionTag.move_to_breakeven:
            return 0.82
        return self._classifier_recovery_probability

    def _classifier_signal_matches(self, signal_tag: ActionTag, classifier_tag: ActionTag) -> bool:
        if signal_tag == classifier_tag:
            return True
        long_family = {ActionTag.enter_long, ActionTag.add}
        short_family = {ActionTag.enter_short, ActionTag.add}
        if signal_tag in long_family and classifier_tag in long_family:
            return True
        if signal_tag in short_family and classifier_tag in short_family:
            return True
        return False

    def _side_for_classifier_tag(self, tag: ActionTag) -> TradeSide | None:
        if tag == ActionTag.enter_long:
            return TradeSide.long
        if tag == ActionTag.enter_short:
            return TradeSide.short
        return None

    def _classifier_prediction_payload(self, prediction: IntentClassifierPrediction) -> dict[str, object]:
        return {
            "tag": prediction.tag.value,
            "confidence": prediction.confidence,
            "action_probability": prediction.action_probability,
            "entry_probability": prediction.entry_probability,
            "management_probability": prediction.management_probability,
            "probabilities": {
                tag.value if isinstance(tag, ActionTag) else str(tag): probability
                for tag, probability in prediction.probabilities.items()
            },
            "thresholds": {
                tag.value if isinstance(tag, ActionTag) else str(tag): threshold
                for tag, threshold in prediction.thresholds.items()
            },
            "model_name": prediction.model_name,
        }

    def _get_state(self, session_id: str, *, mutate_state: bool) -> FlowState:
        state = self._states.get(session_id)
        if state is None:
            if mutate_state:
                state = FlowState()
                self._states[session_id] = state
                return state
            return FlowState()
        if mutate_state:
            return state
        return FlowState(
            pending_entry=PendingEntry(
                side=state.pending_entry.side,
                created_at=state.pending_entry.created_at,
                seed_price=state.pending_entry.seed_price,
                entry_price=state.pending_entry.entry_price,
                evidence_text=state.pending_entry.evidence_text,
            ) if state.pending_entry is not None else None,
            last_management_at=state.last_management_at,
            last_exit_at=state.last_exit_at,
            last_side=state.last_side,
            active_instrument=state.active_instrument,
            active_instrument_at=state.active_instrument_at,
            recent_text=state.recent_text,
            recent_text_at=state.recent_text_at,
            recent_fragments=[
                RecentFragment(
                    text=fragment.text,
                    normalized=fragment.normalized,
                    received_at=fragment.received_at,
                    segment_id=fragment.segment_id,
                )
                for fragment in state.recent_fragments
            ],
            candidate_window=(
                CandidateWindow(
                    opened_at=state.candidate_window.opened_at,
                    updated_at=state.candidate_window.updated_at,
                    probability=state.candidate_window.probability,
                    source=state.candidate_window.source,
                    side_hint=state.candidate_window.side_hint,
                    tag_hint=state.candidate_window.tag_hint,
                    fragments=[
                        RecentFragment(
                            text=fragment.text,
                            normalized=fragment.normalized,
                            received_at=fragment.received_at,
                            segment_id=fragment.segment_id,
                        )
                        for fragment in state.candidate_window.fragments
                    ],
                )
                if state.candidate_window is not None
                else None
            ),
            recent_intent_tag=state.recent_intent_tag,
            recent_intent_side=state.recent_intent_side,
            recent_intent_at=state.recent_intent_at,
            recent_intent_entry_price=state.recent_intent_entry_price,
            recent_intent_stop_price=state.recent_intent_stop_price,
            recent_intent_target_price=state.recent_intent_target_price,
        )

    def _prune_temporal_state(self, state: FlowState, *, received_at: datetime, mutate_state: bool) -> None:
        recent_cutoff = received_at - max(self._candidate_window, self._candidate_preroll_window)
        if state.recent_fragments:
            retained_fragments = [
                fragment for fragment in state.recent_fragments if fragment.received_at >= recent_cutoff
            ][-max(self._candidate_max_fragments * 2, self._candidate_max_fragments):]
            if mutate_state:
                state.recent_fragments = retained_fragments
        if self._active_candidate_window(state, received_at=received_at, mutate_state=mutate_state) is None and mutate_state:
            state.candidate_window = None

    def _remember_recent_fragment(self, state: FlowState, *, context: ParseContext) -> None:
        fragment = RecentFragment(
            text=context.text.strip(),
            normalized=context.normalized,
            received_at=context.received_at,
            segment_id=context.segment_id,
        )
        if state.recent_fragments and state.recent_fragments[-1].segment_id == fragment.segment_id:
            state.recent_fragments[-1] = fragment
        elif state.recent_fragments and state.recent_fragments[-1].normalized == fragment.normalized:
            state.recent_fragments[-1] = fragment
        else:
            state.recent_fragments.append(fragment)
        state.recent_fragments = state.recent_fragments[-max(self._candidate_max_fragments * 2, self._candidate_max_fragments):]

    def _has_active_candidate_window(self, state: FlowState, *, received_at: datetime) -> bool:
        return self._active_candidate_window(state, received_at=received_at, mutate_state=False) is not None

    def _active_candidate_window(
        self,
        state: FlowState,
        *,
        received_at: datetime,
        mutate_state: bool,
    ) -> CandidateWindow | None:
        window = state.candidate_window
        if window is None:
            return None
        if received_at - window.updated_at <= self._candidate_window:
            return window
        if mutate_state:
            state.candidate_window = None
        return None

    def _track_candidate_window(
        self,
        state: FlowState,
        *,
        context: ParseContext,
        assessment: CandidateAssessment,
    ) -> None:
        current_fragment = RecentFragment(
            text=context.text.strip(),
            normalized=context.normalized,
            received_at=context.received_at,
            segment_id=context.segment_id,
        )
        window = self._active_candidate_window(state, received_at=context.received_at, mutate_state=True)
        if window is None:
            seeded_fragments = [
                RecentFragment(
                    text=fragment.text,
                    normalized=fragment.normalized,
                    received_at=fragment.received_at,
                    segment_id=fragment.segment_id,
                )
                for fragment in state.recent_fragments
                if context.received_at - fragment.received_at <= self._candidate_preroll_window
            ]
            if (
                state.recent_intent_at is not None
                and state.recent_intent_tag is not None
                and state.recent_intent_tag not in {ActionTag.setup_long, ActionTag.setup_short}
                and context.received_at - state.recent_intent_at <= self._duplicate_entry_window
            ):
                seeded_fragments = [
                    fragment for fragment in seeded_fragments if fragment.received_at > state.recent_intent_at
                ]
            if not seeded_fragments or seeded_fragments[-1].segment_id != current_fragment.segment_id:
                seeded_fragments.append(current_fragment)
            window = CandidateWindow(
                opened_at=context.received_at,
                updated_at=context.received_at,
                probability=assessment.probability,
                source=assessment.source,
                side_hint=assessment.side_hint,
                tag_hint=assessment.tag_hint,
                fragments=seeded_fragments[-self._candidate_max_fragments:],
            )
            state.candidate_window = window
            return

        window.updated_at = context.received_at
        window.probability = max(window.probability, assessment.probability)
        if assessment.side_hint is not None:
            window.side_hint = assessment.side_hint
        if assessment.tag_hint is not None:
            window.tag_hint = assessment.tag_hint
        if window.fragments and window.fragments[-1].segment_id == current_fragment.segment_id:
            window.fragments[-1] = current_fragment
        else:
            window.fragments.append(current_fragment)
        window.fragments = window.fragments[-self._candidate_max_fragments:]

    def _build_candidate_window_context(
        self,
        *,
        window: CandidateWindow,
        context: ParseContext,
    ) -> ParseContext | None:
        if len(window.fragments) < 2:
            return None
        raw_parts = [fragment.text.strip().rstrip(".!?") for fragment in window.fragments if fragment.text.strip()]
        normalized_parts = [fragment.normalized.strip() for fragment in window.fragments if fragment.normalized.strip()]
        if len(normalized_parts) < 2:
            return None
        raw_text = ". ".join(raw_parts).strip()
        normalized_text = " ".join(normalized_parts).strip()
        if not raw_text or not normalized_text:
            return None
        return ParseContext(
            text=raw_text,
            normalized=normalized_text,
            market_price=context.market_price,
            received_at=context.received_at,
            source_latency_ms=context.source_latency_ms,
            segment_id=context.segment_id,
        )

    def _candidate_window_payload(self, window: CandidateWindow) -> dict[str, object]:
        return {
            "opened_at": window.opened_at.isoformat(),
            "updated_at": window.updated_at.isoformat(),
            "probability": window.probability,
            "source": window.source,
            "side_hint": window.side_hint.value if isinstance(window.side_hint, TradeSide) else window.side_hint,
            "tag_hint": window.tag_hint.value if isinstance(window.tag_hint, ActionTag) else window.tag_hint,
            "fragments": [
                {
                    "text": fragment.text,
                    "normalized": fragment.normalized,
                    "received_at": fragment.received_at.isoformat(),
                    "segment_id": fragment.segment_id,
                }
                for fragment in window.fragments
            ],
        }

    def _get_pending_entry(self, state: FlowState, *, received_at: datetime, mutate_state: bool) -> PendingEntry | None:
        pending = state.pending_entry
        if pending is None:
            return None
        if received_at - pending.created_at <= self._entry_context_window:
            return pending
        if mutate_state:
            state.pending_entry = None
        return None

    def _entry_guard_reason(self, state: FlowState, received_at: datetime) -> str | None:
        if state.last_management_at and received_at - state.last_management_at <= self._entry_guard_window:
            return "recent management cue detected"
        if state.last_exit_at and received_at - state.last_exit_at <= self._entry_guard_window:
            return "recent exit cue detected"
        return None

    def _active_setup_side_hint(self, state: FlowState, *, received_at: datetime) -> TradeSide | None:
        window = self._active_candidate_window(state, received_at=received_at, mutate_state=False)
        if window is None:
            return None
        if window.tag_hint not in {ActionTag.setup_long, ActionTag.setup_short}:
            return None
        return window.side_hint

    def _allows_setup_seed_entry_override(
        self,
        *,
        session: StreamSession,
        state: FlowState,
        context: ParseContext,
        text: str,
    ) -> bool:
        if self._active_setup_side_hint(state, received_at=context.received_at) is None:
            return False
        if not self._is_entry_trigger(text):
            return False
        if self._is_non_actionable_trade_commentary(text, session=session):
            return False
        if self._is_non_actionable_entry_commentary(text):
            return False
        return True

    def _analysis_text(self, state: FlowState, *, text: str, received_at: datetime) -> str:
        if state.recent_text is None or state.recent_text_at is None:
            return text
        if received_at - state.recent_text_at > self._context_stitch_window:
            return text
        return f"{state.recent_text} {text}".strip()

    def _entry_text(self, state: FlowState, *, text: str, received_at: datetime) -> str:
        if state.recent_text is None or state.recent_text_at is None:
            return text
        if received_at - state.recent_text_at > self._context_stitch_window:
            return text
        if self._looks_entry_continuation(text) or self._looks_incomplete_entry_fragment(state.recent_text):
            return f"{state.recent_text} {text}".strip()
        return text

    def _looks_entry_continuation(self, text: str) -> bool:
        return _matches_any(ENTRY_CONTINUATION_PREFIX_PATTERNS, text)

    def _looks_incomplete_entry_fragment(self, text: str) -> bool:
        return _matches_any(ENTRY_INCOMPLETE_SUFFIX_PATTERNS, text)

    def _detect_side(self, text: str) -> TradeSide | None:
        if self._is_non_actionable_entry_commentary(text):
            return None
        if re.search(r"\blong versus\b", text):
            return TradeSide.long
        if re.search(r"\bshort versus\b", text):
            return TradeSide.short
        if _matches_any(DIRECT_LONG_PATTERNS, text):
            return TradeSide.long
        if _matches_any(DIRECT_SHORT_PATTERNS, text):
            return TradeSide.short
        return None

    def _detect_entry_side(
        self,
        text: str,
        *,
        market_price: float | None = None,
        target_price: float | None = None,
    ) -> TradeSide | None:
        if self._is_non_actionable_entry_commentary(text) or self._looks_incomplete_entry_fragment(text):
            return None
        if _matches_any(ENTRY_STRONG_LONG_PATTERNS, text):
            return TradeSide.long
        if _matches_any(ENTRY_STRONG_SHORT_PATTERNS, text):
            return TradeSide.short
        if self._is_contextual_in_trade_phrase(text, side=TradeSide.long):
            return TradeSide.long
        if self._is_contextual_in_trade_phrase(text, side=TradeSide.short):
            return TradeSide.short
        if self._is_actionable_bare_side_phrase(text, side=TradeSide.long):
            return TradeSide.long
        if self._is_actionable_bare_side_phrase(text, side=TradeSide.short):
            return TradeSide.short
        if _matches_any(ENTRY_SEED_PATTERNS, text):
            if re.search(r"\blong\b", text):
                return TradeSide.long
            if re.search(r"\bshort\b", text):
                return TradeSide.short
            directional_side = self._infer_seed_side_from_target_direction(
                text,
                market_price=market_price,
                target_price=target_price,
            )
            if directional_side is not None:
                return directional_side
        return None

    def _infer_seed_side_from_target_direction(
        self,
        text: str,
        *,
        market_price: float | None,
        target_price: float | None,
    ) -> TradeSide | None:
        if market_price is None or target_price is None:
            return None
        if re.search(r"\bup to\b", text) and target_price > market_price:
            return TradeSide.long
        if re.search(r"\bdown to\b", text) and target_price < market_price:
            return TradeSide.short
        return None

    def _is_actionable_bare_side_phrase(self, text: str, *, side: TradeSide) -> bool:
        side_word = "long" if side == TradeSide.long else "short"
        if not re.search(rf"\b{side_word} here\b", text):
            return False
        if re.search(rf"{ENTRY_LEADIN_PATTERN}{side_word} here\b", text):
            return True
        if re.search(rf"\b{side_word} here\b.*{ENTRY_BARE_CONTEXT_PATTERN}", text):
            return True
        if _matches_any(ENTRY_SEED_PATTERNS, text):
            return True
        return False

    def _is_contextual_in_trade_phrase(self, text: str, *, side: TradeSide) -> bool:
        side_word = "long" if side == TradeSide.long else "short"
        if _matches_any(ENTRY_CONTEXTUAL_POSITION_NEGATIVE_PATTERNS, text):
            return False
        anchored = re.search(rf"{ENTRY_LEADIN_PATTERN}in this {side_word}\b", text)
        inline = re.search(rf"\bin this {side_word}\b(?!\s+(?:term|period|time))", text)
        if not anchored and not inline:
            return False
        if any(hint in text for hint in ENTRY_CONTEXTUAL_ACTION_HINTS):
            return True
        if re.search(ENTRY_BARE_CONTEXT_PATTERN, text):
            return True
        return True

    def _is_strong_preview_entry_text(self, text: str, *, side: TradeSide | None) -> bool:
        if side is None:
            return False
        if not any(token in text for token in STRONG_ENTRY_RISK_TOKENS):
            return False
        if side == TradeSide.long:
            return _matches_any(STRONG_SELF_LONG_ENTRY_PATTERNS, text)
        return _matches_any(STRONG_SELF_SHORT_ENTRY_PATTERNS, text)

    def _entry_tag_for_side(self, side: TradeSide, session: StreamSession) -> ActionTag:
        if session.position is not None and session.position.side == side:
            return ActionTag.add
        return ActionTag.enter_long if side == TradeSide.long else ActionTag.enter_short

    def _is_actionable_trim_command(self, text: str) -> bool:
        if not _matches_any(IMPERATIVE_TRIM_PATTERNS, text):
            return False
        if _matches_any(IMPERATIVE_TRIM_NEGATIVE_PATTERNS, text):
            return False
        return True

    def _is_entry_trigger(self, text: str) -> bool:
        if self._is_non_actionable_entry_commentary(text) or self._looks_incomplete_entry_fragment(text):
            return False
        if self._detect_entry_side(text) is not None:
            return True
        return _matches_any(ENTRY_SEED_PATTERNS, text)

    def _find_price(self, text: str, context: ParseContext, patterns: list[str]) -> float | None:
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            raw = match.groupdict().get("price") or match.group(0)
            resolved = _resolve_price(raw, context.market_price)
            if resolved is not None:
                return resolved
        return None

    def _detect_entry_size_hint(self, text: str) -> str | None:
        if "feeler" in text:
            return "one"
        if re.search(r"\b(?:little|small)\s+(?:piece|peace)\b", text):
            return "one"
        if _matches_any(ENTRY_SEED_PATTERNS, text):
            return "one"
        return self._detect_quantity(text)

    def _detect_quantity(self, text: str) -> str | None:
        if "all" in text or "whole position" in text:
            return "all"
        if "most my size" in text:
            return "most"
        if re.search(r"\bhalf\b", text):
            return "half"
        if re.search(r"\bone\b", text):
            return "one"
        return None

    def _score(
        self,
        *,
        tag: ActionTag,
        text: str,
        entry_price: float | None,
        stop_price: float | None,
        target_price: float | None,
        guard_reason: str | None,
    ) -> float:
        """Heuristic confidence score based on action type and extracted context.

        Base scores reflect how unambiguous each action type is in spoken
        language: exit_all (0.93) and breakeven (0.91) use distinctive phrases
        that rarely appear in non-actionable speech, while setups (0.72) are
        inherently tentative.  Bonuses reward concrete price levels that
        corroborate the intent; penalties flag hedging language or guard
        conditions.  Values were calibrated against the reviewed annotation
        corpus (see benchmark_models.py).
        """
        base = {
            ActionTag.enter_long: 0.88,
            ActionTag.enter_short: 0.88,
            ActionTag.add: 0.86,
            ActionTag.trim: 0.87,
            ActionTag.exit_all: 0.93,
            ActionTag.move_stop: 0.88,
            ActionTag.move_to_breakeven: 0.91,
            ActionTag.setup_long: 0.72,
            ActionTag.setup_short: 0.72,
            ActionTag.cancel_setup: 0.8,
        }.get(tag, 0.7)
        if entry_price is not None:
            base += 0.03
        if stop_price is not None:
            base += 0.05
        if target_price is not None:
            base += 0.02
        if "maybe" in text or "could" in text:
            base -= 0.15
        if guard_reason is not None:
            base -= 0.08
        return max(0.0, min(0.99, round(base, 3)))

    def _looks_trade_candidate(self, text: str) -> bool:
        return any(keyword in text for keyword in TRADE_KEYWORDS) or looks_candidate_seed(text)

    def _should_clarify(self, text: str, intent: TradeIntent) -> bool:
        if intent.guard_reason is not None:
            return False
        if intent.confidence < 0.82:
            return True
        if intent.tag in {ActionTag.setup_long, ActionTag.setup_short, ActionTag.commentary}:
            return self._looks_trade_candidate(text)
        return False

    def _should_confirm_with_fallback(self, session: StreamSession, intent: TradeIntent) -> bool:
        if not session.config.enable_ai_fallback or self._fallback is None:
            return False
        fallback_available = getattr(self._fallback, "is_available", None)
        if callable(fallback_available) and not fallback_available():
            return False
        if session.config.execution_mode != ExecutionMode.auto:
            return False
        return intent.tag in {ActionTag.enter_long, ActionTag.enter_short, ActionTag.add}

    def _should_use_extractive_fallback(self, session: StreamSession) -> bool:
        if session.config.execution_mode != ExecutionMode.review:
            return False
        fallback_available = getattr(self._fallback, "is_available", None)
        if callable(fallback_available) and not fallback_available():
            return False
        return True

    async def _confirm_with_fallback(
        self,
        *,
        session: StreamSession,
        segment: TranscriptSegment,
        context: ParseContext,
        intent: TradeIntent,
        state_before: FlowState,
    ) -> ConfirmationDecision:
        confirm_intent = getattr(self._fallback, "confirm_intent", None)
        if confirm_intent is None:
            self._restore_state_after_veto(session.id, state_before)
            return ConfirmationDecision(approved=False, reason="gemini confirmation unavailable")

        confirmation_context = self._entry_text(
            state_before,
            text=context.normalized,
            received_at=context.received_at,
        )
        try:
            confirmation = await asyncio.wait_for(
                confirm_intent(
                    session=session,
                    segment=segment,
                    proposed_intent=intent,
                    context_text=confirmation_context,
                ),
                timeout=self._fallback_confirmation_timeout_s,
            )
        except asyncio.TimeoutError:
            confirmation = GeminiConfirmation(confirmed=False, reason="gemini confirmation timed out", system_failure=True)
        except Exception:
            confirmation = GeminiConfirmation(confirmed=False, reason="gemini confirmation unavailable", system_failure=True)

        if confirmation is not None and getattr(confirmation, "confirmed", False):
            return ConfirmationDecision(approved=True)

        reason = getattr(confirmation, "reason", None) or "Gemini did not confirm the entry."
        if self._should_fail_open_entry_confirmation(intent=intent, text=confirmation_context, confirmation=confirmation):
            return ConfirmationDecision(
                approved=True,
                reason=f"{reason}; executed on strong rule signal.",
                fail_open_applied=True,
            )

        if confirmation is None or not getattr(confirmation, "confirmed", False):
            self._restore_state_after_veto(session.id, state_before)
            return ConfirmationDecision(approved=False, reason=reason)
        return ConfirmationDecision(approved=True)

    def _restore_state_after_veto(self, session_id: str, state_before: FlowState) -> None:
        state = self._states.get(session_id)
        if state is None:
            return
        state.pending_entry = (
            PendingEntry(
                side=state_before.pending_entry.side,
                created_at=state_before.pending_entry.created_at,
                seed_price=state_before.pending_entry.seed_price,
                entry_price=state_before.pending_entry.entry_price,
                evidence_text=state_before.pending_entry.evidence_text,
            )
            if state_before.pending_entry is not None
            else None
        )
        state.last_management_at = state_before.last_management_at
        state.last_exit_at = state_before.last_exit_at
        state.last_side = state_before.last_side
        state.recent_intent_tag = state_before.recent_intent_tag
        state.recent_intent_side = state_before.recent_intent_side
        state.recent_intent_at = state_before.recent_intent_at
        state.recent_intent_entry_price = state_before.recent_intent_entry_price
        state.recent_intent_stop_price = state_before.recent_intent_stop_price
        state.recent_intent_target_price = state_before.recent_intent_target_price

    def _should_fail_open_entry_confirmation(
        self,
        *,
        intent: TradeIntent,
        text: str,
        confirmation: GeminiConfirmation | None,
    ) -> bool:
        if intent.tag not in {ActionTag.enter_long, ActionTag.enter_short}:
            return False
        if intent.guard_reason is not None or intent.stop_price is None or intent.confidence < 0.9:
            return False
        if confirmation is not None and not getattr(confirmation, "system_failure", False):
            return False
        if not any(token in text for token in STRONG_ENTRY_RISK_TOKENS):
            return False
        if intent.side == TradeSide.long:
            return _matches_any(STRONG_SELF_LONG_ENTRY_PATTERNS, text)
        if intent.side == TradeSide.short:
            return _matches_any(STRONG_SELF_SHORT_ENTRY_PATTERNS, text)
        return False

    def _is_non_actionable_entry_commentary(self, text: str) -> bool:
        return _matches_any(NON_ACTIONABLE_ENTRY_PATTERNS, text)

    def _is_non_actionable_trade_commentary(self, text: str, *, session: StreamSession) -> bool:
        if _matches_any(NON_ACTIONABLE_TRADE_PATTERNS, text):
            return True
        if not self._advisory_intents_enabled(session) and _matches_any(AUTO_ONLY_NON_ACTIONABLE_TRADE_PATTERNS, text):
            return True
        return False

    def _advisory_intents_enabled(self, session: StreamSession) -> bool:
        return session.config.execution_mode == ExecutionMode.review

    def _mentions_session_instrument(self, text: str, session: StreamSession) -> bool:
        aliases = _session_symbol_aliases(session)
        return any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases)

    def _find_foreign_instrument(self, text: str, session: StreamSession) -> str | None:
        for pattern in FOREIGN_CONTEXT_PATTERNS:
            if re.search(pattern, text):
                return "swing_context"
        session_aliases = _session_symbol_aliases(session)
        inline_trade_instrument = _find_inline_trade_instrument(text, session_aliases)
        if inline_trade_instrument is not None:
            return inline_trade_instrument
        for token in FOREIGN_INSTRUMENT_TOKENS:
            if token in session_aliases:
                continue
            if re.search(rf"\b{re.escape(token)}\b", text):
                return token
        return None

    def _active_foreign_instrument(self, state: FlowState, *, received_at: datetime) -> str | None:
        if state.active_instrument is None or state.active_instrument_at is None:
            return None
        if received_at - state.active_instrument_at <= self._instrument_context_window:
            return state.active_instrument
        state.active_instrument = None
        state.active_instrument_at = None
        return None

    def _should_suppress_duplicate_intent(self, state: FlowState, *, context: ParseContext, intent: TradeIntent) -> bool:
        if state.recent_intent_tag is None or state.recent_intent_at is None:
            return False
        age = context.received_at - state.recent_intent_at
        if age <= self._duplicate_intent_window:
            if intent.tag == state.recent_intent_tag and intent.side == state.recent_intent_side and self._same_prices(
                intent.entry_price,
                state.recent_intent_entry_price,
                intent.stop_price,
                state.recent_intent_stop_price,
                intent.target_price,
                state.recent_intent_target_price,
            ):
                return True
        if age <= self._duplicate_entry_window:
            if (
                self._is_entry_family(intent.tag)
                and self._is_entry_family(state.recent_intent_tag)
                and intent.side == state.recent_intent_side
                and self._same_prices(
                    intent.entry_price,
                    state.recent_intent_entry_price,
                    intent.stop_price,
                    state.recent_intent_stop_price,
                    intent.target_price,
                    state.recent_intent_target_price,
                )
                and self._looks_duplicate_entry_followup(context.normalized)
            ):
                return True
        return False

    def _remember_intent(self, state: FlowState, *, context: ParseContext, intent: TradeIntent) -> None:
        state.recent_intent_tag = intent.tag
        state.recent_intent_side = intent.side
        state.recent_intent_at = context.received_at
        state.recent_intent_entry_price = intent.entry_price
        state.recent_intent_stop_price = intent.stop_price
        state.recent_intent_target_price = intent.target_price

    def _is_entry_family(self, tag: ActionTag) -> bool:
        return tag in {ActionTag.enter_long, ActionTag.enter_short, ActionTag.add}

    def _looks_duplicate_entry_followup(self, text: str) -> bool:
        if any(token in text for token in ("versus", "stop", "reclaim", "breakeven", "target")):
            return False
        if _matches_any(ENTRY_SEED_PATTERNS, text):
            return True
        return any(
            re.search(pattern, text)
            for pattern in [
                r"\bjust a feeler on\b",
                r"\bwe just got piece on\b",
                r"\bwe just got peace on\b",
                r"\bput(?:ting)? (?:something|a little|a small)?\s*(?:piece|peace)\s+on\b",
            ]
        )

    def _is_retrospective_recovery_text(self, text: str) -> bool:
        return _matches_any(RETROSPECTIVE_RECOVERY_NEGATIVE_PATTERNS, text)

    def _same_prices(
        self,
        left_entry: float | None,
        right_entry: float | None,
        left_stop: float | None,
        right_stop: float | None,
        left_target: float | None,
        right_target: float | None,
    ) -> bool:
        return (
            _price_equal(left_entry, right_entry)
            and _price_equal(left_stop, right_stop)
            and _price_equal(left_target, right_target)
        )


def _normalize(text: str) -> str:
    lowered = apply_trading_asr_corrections(text)
    lowered = lowered.replace("-", " ")
    lowered = re.sub(r"\bpoint of\b", "point", lowered)
    lowered = re.sub(r"[^\w\s\.]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _compact_repeated_trade_text(text: str) -> str:
    compacted = re.sub(r"\s+", " ", text).strip()
    if not compacted:
        return compacted
    chunks = [part.strip() for part in re.findall(r"[^.!?]+[.!?]?", compacted) if part.strip()]
    if not chunks:
        return compacted

    filtered: list[str] = []
    last_normalized = ""
    repeat_count = 0
    for chunk in chunks:
        normalized_chunk = _normalize(chunk)
        if not normalized_chunk:
            continue
        if normalized_chunk == last_normalized:
            repeat_count += 1
        else:
            last_normalized = normalized_chunk
            repeat_count = 1
        max_repeats = 1 if len(normalized_chunk.split()) <= 4 else 2
        if repeat_count > max_repeats:
            continue
        filtered.append(chunk)
    if not filtered:
        return compacted
    return " ".join(filtered)


def _coerce_received_at(segment: TranscriptSegment) -> datetime:
    received_at = segment.received_at
    if received_at.tzinfo is None:
        return received_at.replace(tzinfo=UTC)
    return received_at


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _price_equal(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return abs(left - right) < 0.01


def _session_symbol_aliases(session: StreamSession) -> set[str]:
    symbol = session.market.symbol or session.config.symbol
    root = _symbol_root(symbol)
    if root is None:
        return set()
    return SESSION_SYMBOL_ALIASES.get(root, {root.lower()})


def _find_inline_trade_instrument(text: str, session_aliases: set[str]) -> str | None:
    excluded = {
        "all",
        "back",
        "front",
        "here",
        "it",
        "my",
        "one",
        "that",
        "the",
        "there",
        "this",
    }
    patterns = [
        r"\b(?:i m|i am|was|were)\s+(?:long|short)\s+in\s+(?P<instrument>[a-z][a-z0-9]{2,})\b",
        r"\b(?:started\s+getting|get(?:ting)?|got)\s+(?:long|short)\s+in\s+(?P<instrument>[a-z][a-z0-9]{2,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        instrument = match.group("instrument")
        if instrument in excluded or instrument in session_aliases:
            continue
        return instrument
    return None


def _symbol_root(symbol: str | None) -> str | None:
    if not symbol:
        return None
    match = re.search(r"[A-Za-z]{1,5}", symbol)
    if match is None:
        return None
    return match.group(0).upper()


def _resolve_price(raw_text: str, market_price: float | None) -> float | None:
    raw_text = raw_text.strip().lower()
    raw_text = raw_text.replace("-", " ")
    raw_text = re.sub(r"(?<=\d),(?=\d)", "", raw_text)
    if "three quarter" in raw_text:
        raw_text = raw_text.replace("three quarter", "")
        quarter = 0.75
    elif "quarter" in raw_text:
        raw_text = raw_text.replace("quarter", "")
        quarter = 0.25
    elif "half" in raw_text:
        raw_text = raw_text.replace("half", "")
        quarter = 0.5
    else:
        quarter = 0.0

    numeric_candidates = _extract_numeric_candidates(raw_text)
    if not numeric_candidates:
        return None

    resolved_candidates: list[float] = []
    for value in numeric_candidates:
        if value >= 1_000:
            resolved_candidates.append(round(value + quarter, 2))
            continue
        if market_price is None:
            resolved_candidates.append(round(value + quarter, 2))
            continue
        resolved_candidates.extend(_expand_shorthand_candidates(value, market_price, quarter))

    if not resolved_candidates:
        return None
    if market_price is None:
        return resolved_candidates[0]
    return round(min(resolved_candidates, key=lambda price: abs(price - market_price)), 2)


def _extract_numeric_candidates(raw_text: str) -> list[float]:
    candidates: list[float] = []

    match = PRICE_TOKEN_PATTERN.search(raw_text)
    if match:
        token = match.group("num")
        try:
            candidates.append(float(token))
        except ValueError:
            pass

    for candidate in _extract_word_number_candidates(raw_text):
        candidates.append(float(candidate))

    deduped: list[float] = []
    seen: set[float] = set()
    for candidate in candidates:
        rounded = round(float(candidate), 2)
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(rounded)
    return deduped


def _extract_word_number_candidates(raw_text: str) -> list[int]:
    cleaned_tokens = [re.sub(r"[^a-z]", "", token) for token in raw_text.split()]
    tokens = [
        token
        for token in cleaned_tokens
        if token and (token in _NUMBER_WORDS or token in _SCALE_WORDS or token in _NUMBER_FILLER_WORDS)
    ]
    if not tokens:
        return []

    candidates: list[int] = []

    conventional = _parse_number_words(tokens)
    if conventional is not None:
        candidates.append(conventional)

    shorthand_groups = _extract_shorthand_groups(tokens)
    for group_values in shorthand_groups:
        candidates.extend(_build_grouped_number_candidates(group_values))

    deduped: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate < 0 or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _parse_number_words(tokens: list[str]) -> int | None:
    total = 0
    current = 0
    consumed = False

    for token in tokens:
        if token in _NUMBER_FILLER_WORDS:
            continue
        if token in _GROUP_SEPARATOR_WORDS:
            return None
        if token in _NUMBER_WORDS:
            current += _NUMBER_WORDS[token]
            consumed = True
            continue
        scale = _SCALE_WORDS.get(token)
        if scale is None:
            return None
        consumed = True
        if current == 0:
            current = 1
        if scale == 100:
            current *= scale
            continue
        total += current * scale
        current = 0

    if not consumed:
        return None
    return total + current


def _extract_shorthand_groups(tokens: list[str]) -> list[list[int]]:
    compact_tokens = [token for token in tokens if token not in _NUMBER_FILLER_WORDS]
    if not compact_tokens or any(token in _SCALE_WORDS for token in compact_tokens):
        return []
    if len(compact_tokens) > 4:
        return []

    partitions: list[list[int]] = []

    def walk(index: int, current_groups: list[int]) -> None:
        if index >= len(compact_tokens):
            if len(current_groups) >= 2:
                partitions.append(current_groups.copy())
            return

        token = compact_tokens[index]
        if token in _GROUP_SEPARATOR_WORDS:
            walk(index + 1, current_groups)
            return

        for end in range(index + 1, min(len(compact_tokens), index + 2) + 1):
            group_tokens = compact_tokens[index:end]
            if any(part in _GROUP_SEPARATOR_WORDS for part in group_tokens):
                break
            value = _parse_number_words(group_tokens)
            if value is None or value >= 100:
                continue
            current_groups.append(value)
            walk(end, current_groups)
            current_groups.pop()

    walk(0, [])
    return partitions


def _build_grouped_number_candidates(groups: list[int]) -> list[int]:
    if len(groups) < 2:
        return []

    candidates: set[int] = set()

    def build(index: int, current: str) -> None:
        if index >= len(groups):
            try:
                candidates.add(int(current))
            except ValueError:
                return
            return

        value = groups[index]
        parts = {str(value)}
        if index > 0:
            parts.add(str(value).zfill(2))
            parts.add(str(value).zfill(3))
        for part in parts:
            build(index + 1, current + part)

    build(1, str(groups[0]))
    return sorted(candidates)


def _expand_shorthand_candidates(value: float, market_price: float, quarter: float) -> list[float]:
    market_int = int(round(market_price))
    integer_value = int(value)
    width = len(str(abs(integer_value)).split(".")[0])
    modulus = 10 ** min(width, 3)
    candidates: list[float] = []

    for offset in range(-5, 6):
        base = market_int + offset * 100
        candidate = base - (base % modulus) + integer_value
        candidates.append(candidate + quarter)
        if candidate - modulus > 0:
            candidates.append(candidate - modulus + quarter)
        candidates.append(candidate + modulus + quarter)

    deduped: list[float] = []
    seen: set[float] = set()
    for candidate in candidates:
        rounded = round(candidate, 2)
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(rounded)
    return deduped
