"""Labeled accuracy benchmark for the catalyst-detection engine.

This is the measurement the company cares about: how accurately do we detect
catalysts? Each case is a realistic biotech headline labeled with the correct
event type, whether it is a *fired* catalyst (vs. forward-looking), and the
expected market direction. The suite computes precision / recall / F1 and asserts
minimum bars so accuracy can never silently regress.

Run:  pytest tests/test_detection.py -s
"""
from __future__ import annotations

from app.services.detection import detect

# (text, event_type, is_fired_catalyst, direction)
# is_fired_catalyst=False means a forward-looking announcement or a non-event.
CASES: list[tuple[str, str, bool, str]] = [
    # --- trial readouts (fired) ---
    ("Acme Bio announces topline Phase 3 results: GLP-1 trial met the primary endpoint", "trial_readout", True, "positive"),
    ("Zenith reports its pivotal trial failed to meet the primary endpoint", "trial_readout", True, "negative"),
    ("Nova's Phase 2 study achieved statistically significant improvement in PFS", "trial_readout", True, "positive"),
    ("Helix announces Phase 3 readout; drug did not meet primary endpoint", "trial_readout", True, "negative"),
    ("Orion presents late-breaking Phase 3 data at ASCO showing superior response", "trial_readout", True, "positive"),
    # --- FDA decisions (fired) ---
    ("FDA approves Madrigal's resmetirom for NASH", "fda_decision", True, "positive"),
    ("Sarepta receives complete response letter from the FDA", "fda_decision", True, "negative"),
    ("FDA grants accelerated approval to lecanemab", "fda_decision", True, "positive"),
    ("Company announces FDA rejected its new drug application", "fda_decision", True, "negative"),
    # --- licensing deals (fired) ---
    ("BioCo and Pharma enter exclusive worldwide license agreement worth $2B", "licensing_deal", True, "neutral"),
    ("Genna signs collaboration and license agreement with $150M upfront payment", "licensing_deal", True, "positive"),
    # --- acquisitions (fired) ---
    ("Pfizer to acquire Seagen in $43 billion all-cash transaction", "acquisition", True, "positive"),
    ("Amgen announces definitive merger agreement to acquire Horizon Therapeutics", "acquisition", True, "positive"),
    # --- forward-looking (NOT fired catalysts) ---
    ("Acme expects to report topline Phase 3 data in Q4 2026", "trial_readout", False, "neutral"),
    ("Company plans to file an NDA with the FDA in the second half of 2026", "fda_decision", False, "neutral"),
    ("Nova is on track to initiate its pivotal trial later this year", "trial_readout", False, "neutral"),
    ("Management reiterated guidance for a PDUFA date in 2026", "fda_decision", False, "neutral"),
    # --- non-catalyst noise (NOT catalysts) ---
    ("Biotech CEO to present at the Jefferies Healthcare Conference", "other_catalyst", False, "neutral"),
    ("Company appoints new Chief Financial Officer", "other_catalyst", False, "neutral"),
    ("Quarterly earnings call scheduled for next Tuesday", "other_catalyst", False, "neutral"),
    # --- false-positive traps (short tokens must NOT match inside words) ---
    ("Company updates corporate agenda and establishes new committee", "other_catalyst", False, "neutral"),
]


def _metrics(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def test_detection_accuracy_benchmark():
    type_correct = 0
    type_total = 0
    # fired-catalyst detection confusion counts
    tp = fp = tn = fn = 0
    dir_correct = 0
    dir_total = 0

    for text, exp_type, exp_fired, exp_dir in CASES:
        d = detect(text)

        # Event-type accuracy (only graded on fired catalysts where type is meaningful)
        if exp_fired:
            type_total += 1
            if d.event_type == exp_type:
                type_correct += 1

        # Fired-catalyst detection (precision/recall of "is this a real catalyst?")
        pred_fired = d.is_catalyst
        if exp_fired and pred_fired:
            tp += 1
        elif exp_fired and not pred_fired:
            fn += 1
        elif not exp_fired and pred_fired:
            fp += 1
        else:
            tn += 1

        # Direction accuracy (graded on fired catalysts)
        if exp_fired:
            dir_total += 1
            if d.direction == exp_dir:
                dir_correct += 1

    precision, recall, f1 = _metrics(tp, fp, fn)
    type_acc = type_correct / type_total if type_total else 0.0
    dir_acc = dir_correct / dir_total if dir_total else 0.0

    print("\n=== Catalyst-detection benchmark ===")
    print(f"Cases:               {len(CASES)}")
    print(f"Event-type accuracy: {type_acc:.0%} ({type_correct}/{type_total})")
    print(f"Fired-catalyst precision/recall/F1: {precision:.0%} / {recall:.0%} / {f1:.2f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Direction accuracy:  {dir_acc:.0%} ({dir_correct}/{dir_total})")

    # Accuracy bars (regression guards).
    assert type_acc >= 0.90, f"event-type accuracy too low: {type_acc:.0%}"
    assert precision >= 0.85, f"catalyst precision too low: {precision:.0%}"
    assert recall >= 0.90, f"catalyst recall too low: {recall:.0%}"
    assert dir_acc >= 0.85, f"direction accuracy too low: {dir_acc:.0%}"


def test_forward_looking_is_flagged():
    d = detect("Acme expects to report topline Phase 3 data in Q4 2026")
    assert d.is_forward_looking is True
    assert d.is_catalyst is False


def test_short_token_false_positive_guard():
    # "agenda" must not trigger an NDA / FDA decision.
    d = detect("Company updates corporate agenda and establishes new committee")
    assert d.event_type == "other_catalyst"
    assert d.is_catalyst is False


def test_negated_positive_is_negative():
    d = detect("Company announces drug was not approved by the FDA")
    assert d.direction == "negative"
