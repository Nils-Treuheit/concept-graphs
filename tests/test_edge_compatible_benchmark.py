"""Small deterministic checks for the paired Replica benchmark helpers."""

from pathlib import Path

from conceptgraph.scripts.benchmark_edge_compatible import (
    edge_metrics,
    normalize_edges,
    pipeline_label_list,
    select_evenly,
    word_f1,
)


def test_pipeline_label_ids_match_saved_detection_labels():
    assert pipeline_label_list(["sofa chair 0", "stool 5", "invalid"]) == [
        "0: sofa chair",
        "5: stool",
    ]


def test_edge_agreement_is_order_independent_and_reports_set_scores():
    baseline = [("1", "on top of", "2"), ("3", "under", "4")]
    candidate = [("3", "under", "4"), ("5", "on top of", "6")]
    assert normalize_edges(candidate) == {
        ("3", "under", "4"),
        ("5", "on top of", "6"),
    }
    scores = edge_metrics(candidate, baseline)
    assert scores["agreement_precision"] == 0.5
    assert scores["agreement_recall"] == 0.5
    assert scores["agreement_f1"] == 0.5


def test_word_f1_and_even_frame_selection_are_deterministic():
    assert word_f1("A round beige stool", "round beige stool") == 6 / 7
    items = [Path(f"frame{i:06}.pkl.gz") for i in range(10)]
    assert [p.name for p in select_evenly(items, 3)] == [
        "frame000000.pkl.gz",
        "frame000004.pkl.gz",
        "frame000009.pkl.gz",
    ]
