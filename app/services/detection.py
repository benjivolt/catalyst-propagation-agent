from __future__ import annotations

import re
from dataclasses import dataclass, field

STRONG, MEDIUM, WEAK = 1.0, 0.55, 0.25

SIGNALS: dict[str, list[tuple[str, float]]] = {
    "trial_readout": [
        (r"met the primary endpoint", STRONG),
        (r"achieved the primary endpoint", STRONG),
        (r"primary endpoint was met", STRONG),
        (r"failed to meet (the )?primary endpoint", STRONG),
        (r"did not meet (the )?primary endpoint", STRONG),
        (r"missed (the )?primary endpoint", STRONG),
        (r"topline (results|data)", STRONG),
        (r"statistically significant", STRONG),
        (r"pivotal (trial|study|data)", STRONG),
        (r"phase [23] (results|data|trial)", STRONG),
        (r"met (its|all) (key |co-)?(secondary )?endpoints?", STRONG),
        (r"readout", MEDIUM),
        (r"interim analysis", MEDIUM),
        (r"primary endpoint", MEDIUM),
        (r"phase [23]", MEDIUM),
        (r"late-breaking", MEDIUM),
        (r"topline", MEDIUM),
        (r"clinical data", WEAK),
        (r"phase 1", WEAK),
        (r"abstract", WEAK),
        (r"endpoint", WEAK),
        (r"trial results", WEAK),
    ],
    "fda_decision": [
        (r"fda approve[ds]?", STRONG),
        (r"approved by the fda", STRONG),
        (r"(received|grants?|granted) (fda )?approval", STRONG),
        (r"complete response letter", STRONG),
        (r"fda reject(s|ed)?", STRONG),
        (r"accelerated approval", STRONG),
        (r"(full|traditional) approval", STRONG),
        (r"\bcrl\b", STRONG),
        (r"pdufa", MEDIUM),
        (r"priority review", MEDIUM),
        (r"breakthrough therapy", MEDIUM),
        (r"advisory committee", MEDIUM),
        (r"label expansion", MEDIUM),
        (
            r"(accepted|acceptance) (the |of )?(\w+ )?(bla|nda|snda|biologics license application|new "
            r"drug application)",
            MEDIUM,
        ),
        (r"fast track", MEDIUM),
        (r"\bbla\b", WEAK),
        (r"\bnda\b", WEAK),
        (r"\bsnda\b", WEAK),
        (r"\bfda\b", WEAK),
    ],
    "licensing_deal": [
        (r"licens(e|ing) agreement", STRONG),
        (r"exclusive (worldwide )?licens", STRONG),
        (r"collaboration (and license )?agreement", STRONG),
        (r"strategic partnership", STRONG),
        (r"option (and license )?agreement", MEDIUM),
        (r"co-development", MEDIUM),
        (r"upfront payment", MEDIUM),
        (r"milestone payments?", MEDIUM),
        (r"royalt(y|ies)", MEDIUM),
        (r"exclusive rights", MEDIUM),
        (r"licens(e|ing)", WEAK),
        (r"collaboration", WEAK),
        (r"partnership", WEAK),
    ],
    "acquisition": [
        (r"to acquire", STRONG),
        (r"acquires?", STRONG),
        (r"to be acquired", STRONG),
        (r"merger agreement", STRONG),
        (r"definitive (merger )?agreement to acquire", STRONG),
        (r"tender offer", STRONG),
        (r"all-cash transaction", STRONG),
        (r"acquisition", MEDIUM),
        (r"merger", MEDIUM),
        (r"buyout", MEDIUM),
        (r"definitive agreement", MEDIUM),
    ],
}

PRIORITY = ["fda_decision", "acquisition", "trial_readout", "licensing_deal"]

FORWARD_MARKERS = [
    r"plans? to",
    r"plan to",
    r"expects? to",
    r"expected (in|during|by|to)",
    r"anticipates?",
    r"intends? to",
    r"on track to",
    r"will (report|present|announce|file|initiate)",
    r"to be (reported|presented|announced)",
    r"upcoming",
    r"later this (year|quarter)",
    r"in the (coming|second half|first half)",
    r"\bguidance\b",
    r"is expected",
    r"projected",
    r"scheduled to",
    r"targeting (a )?(\d{4}|mid|late|early)",
    r"data (are |is )?expected",
    r"readout (is )?expected",
    r"by year-end",
    r"\bq[1-4] \d{4}\b",
    r"in \d{4}",
]

COMPLETION_MARKERS = [
    r"announced (today|that)",
    r"reported (today|that|topline)",
    r"\bmet\b",
    r"\bachieved\b",
    r"\bfailed\b",
    r"\bapproved\b",
    r"\brejected\b",
    r"received",
    r"completed",
    r"presented (data|results)",
    r"today announced",
    r"has (acquired|completed|received)",
]

POSITIVE_MARKERS = [
    r"met the primary endpoint",
    r"achieved the primary endpoint",
    r"statistically significant",
    r"\bapprove[ds]\b",
    r"\bapproval\b",
    r"accelerated approval",
    r"breakthrough therapy",
    r"priority review",
    r"acquir(e|es|ed|ing)",
    r"all-cash",
    r"\bbuyout\b",
    r"upfront payment",
    r"\bvalidated\b",
    r"superior",
    r"first (treatment|approval)",
    r"grant(s|ed)?",
    r"acceptance",
    r"positive (results|data|topline)",
    r"clinically meaningful",
]
NEGATIVE_MARKERS = [
    r"failed to meet",
    r"did not meet",
    r"missed (the )?primary",
    r"complete response letter",
    r"\bcrl\b",
    r"reject(s|ed)",
    r"terminated",
    r"discontinued",
    r"safety signal",
    r"serious adverse",
    r"clinical hold",
    r"inferior",
    r"\bdelay(s|ed)?\b",
    r"did not (achieve|demonstrate)",
    r"disappoint",
    r"\bhalt(s|ed)?\b",
]
NEGATORS = [r"\bnot\b", r"\bno\b", r"\bfail(s|ed|ure)?\b", r"\bwithout\b", r"\bunable\b", r"\bdenied\b"]


def _compile(source: str) -> re.Pattern:
    pat = source
    if not pat.startswith(r"\b") and pat[0].isalnum():
        pat = r"\b" + pat
    if not pat.endswith(r"\b") and pat[-1].isalnum():
        pat = pat + r"\b"
    return re.compile(pat, re.IGNORECASE)


_COMPILED: dict[str, list[tuple[re.Pattern, float]]] = {
    et: [(_compile(src), w) for src, w in sigs] for et, sigs in SIGNALS.items()
}
_FORWARD = [_compile(m) for m in FORWARD_MARKERS]
_COMPLETION = [_compile(m) for m in COMPLETION_MARKERS]
_POSITIVE = [_compile(m) for m in POSITIVE_MARKERS]
_NEGATIVE = [_compile(m) for m in NEGATIVE_MARKERS]
_NEGATORS = [_compile(m) for m in NEGATORS]


@dataclass
class CatalystDetection:
    event_type: str
    confidence: float
    is_catalyst: bool
    is_forward_looking: bool
    direction: str
    direction_confidence: float
    signals: list[str] = field(default_factory=list)
    type_scores: dict[str, float] = field(default_factory=dict)


MIN_CATALYST_CONFIDENCE = 0.35
FORWARD_DISCOUNT = 0.35


def _noisy_or(weights: list[float]) -> float:
    p = 1.0
    for w in weights:
        p *= 1.0 - max(0.0, min(1.0, w))
    return 1.0 - p


def _text(title: str, excerpt: str = "") -> str:
    return f" {title or ''} {excerpt or ''} ".strip()


def detect(title: str, excerpt: str = "") -> CatalystDetection:
    text = _text(title, excerpt)

    type_scores: dict[str, float] = {}
    matched: dict[str, list[str]] = {}
    for event_type, patterns in _COMPILED.items():
        weights: list[float] = []
        hits: list[str] = []
        for pat, w in patterns:
            if pat.search(text):
                weights.append(w)
                hits.append(pat.pattern)
        if weights:
            type_scores[event_type] = round(_noisy_or(weights), 4)
            matched[event_type] = hits

    if not type_scores:
        generic = bool(re.search(r"\b(fda|phase|trial|acquire|licen[cs]e|endpoint|approval)\b", text, re.IGNORECASE))
        direction, dconf = infer_direction(title, excerpt)
        return CatalystDetection(
            event_type="other_catalyst",
            confidence=0.15 if generic else 0.0,
            is_catalyst=False,
            is_forward_looking=_is_forward(text),
            direction=direction,
            direction_confidence=dconf,
            signals=[],
            type_scores={},
        )

    best_score = max(type_scores.values())
    tied = [et for et, s in type_scores.items() if abs(s - best_score) < 1e-9]
    best_type = min(tied, key=lambda et: PRIORITY.index(et) if et in PRIORITY else 99)

    confidence = type_scores[best_type]

    forward = _is_forward(text)
    completed = any(p.search(text) for p in _COMPLETION)
    if forward and not completed:
        confidence = round(confidence * FORWARD_DISCOUNT, 4)

    direction, dconf = infer_direction(title, excerpt)
    forward_flag = forward and not completed
    is_catalyst = confidence >= MIN_CATALYST_CONFIDENCE and not forward_flag

    return CatalystDetection(
        event_type=best_type,
        confidence=confidence,
        is_catalyst=is_catalyst,
        is_forward_looking=forward_flag,
        direction=direction,
        direction_confidence=dconf,
        signals=matched.get(best_type, []),
        type_scores=type_scores,
    )


def _is_forward(text: str) -> bool:
    return any(p.search(text) for p in _FORWARD)


def _has_negated(text: str, pattern: re.Pattern, window: int = 24) -> bool:
    for m in pattern.finditer(text):
        start = max(0, m.start() - window)
        prefix = text[start : m.start()]
        if any(neg.search(prefix) for neg in _NEGATORS):
            return True
    return False


def infer_direction(title: str, excerpt: str = "") -> tuple[str, float]:
    text = _text(title, excerpt).lower()
    pos = 0
    neg = 0
    for p in _POSITIVE:
        if p.search(text):
            if _has_negated(text, p):
                neg += 1
            else:
                pos += 1
    for p in _NEGATIVE:
        if p.search(text):
            neg += 1
    if pos > neg:
        return "positive", min(0.5 + pos * 0.1, 0.85)
    if neg > pos:
        return "negative", min(0.5 + neg * 0.1, 0.85)
    return "neutral", 0.4


def classify_catalyst(title: str, excerpt: str = "") -> str:
    return detect(title, excerpt).event_type


def is_likely_catalyst(title: str, excerpt: str = "") -> bool:
    return detect(title, excerpt).is_catalyst


def detection_confidence(title: str, excerpt: str = "") -> float:
    return detect(title, excerpt).confidence
