"""Benchmark loading and metrics for lexical-relation classifiers."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from lexical_analyzer import RELATIONS, analyze_with_wordnet

DATASET_PATH = Path(__file__).parent / "data" / "evaluation_pairs.csv"


def load_benchmark(path: Path = DATASET_PATH) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as dataset_file:
        rows = list(csv.DictReader(dataset_file))
    required_columns = {"left", "right", "expected_relation"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(f"Benchmark must contain rows with columns: {', '.join(sorted(required_columns))}.")
    for row in rows:
        if row["expected_relation"] not in RELATIONS:
            raise ValueError(f"Unsupported benchmark relation: {row['expected_relation']}")
    return rows


def calculate_metrics(expected: Iterable[str], predicted: Iterable[str]) -> dict[str, object]:
    expected_labels = list(expected)
    predicted_labels = list(predicted)
    if len(expected_labels) != len(predicted_labels) or not expected_labels:
        raise ValueError("Expected and predicted labels must have the same non-zero length.")

    labels = sorted(set(expected_labels) | set(predicted_labels))
    per_class: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
    for label in labels:
        true_positive = sum(
            actual == label and guess == label
            for actual, guess in zip(expected_labels, predicted_labels)
        )
        false_positive = sum(
            actual != label and guess == label
            for actual, guess in zip(expected_labels, predicted_labels)
        )
        false_negative = sum(
            actual == label and guess != label
            for actual, guess in zip(expected_labels, predicted_labels)
        )
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(actual == label for actual in expected_labels),
        }
        f1_scores.append(f1)

    correct = sum(actual == guess for actual, guess in zip(expected_labels, predicted_labels))
    return {
        "accuracy": correct / len(expected_labels),
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "per_class": per_class,
        "count": len(expected_labels),
    }


def evaluate_classifier(
    rows: list[dict[str, str]],
    classifier: Callable[[str, str], str] | None = None,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    classify = classifier or (
        lambda left, right: analyze_with_wordnet(left, right).primary_relation
    )
    results = [
        {
            **row,
            "predicted_relation": classify(row["left"], row["right"]),
        }
        for row in rows
    ]
    metrics = calculate_metrics(
        (row["expected_relation"] for row in results),
        (row["predicted_relation"] for row in results),
    )
    return metrics, results


def compare_classifiers(
    wordnet_rows: list[dict[str, str]],
    llm_rows: list[dict[str, str]],
) -> dict[str, object]:
    """Compare two classifiers on exactly the same ordered benchmark rows."""
    if len(wordnet_rows) != len(llm_rows) or not wordnet_rows:
        raise ValueError("Both classifiers must have results for the same non-zero rows.")

    paired_rows: list[dict[str, str | bool]] = []
    for index, (wordnet_row, llm_row) in enumerate(zip(wordnet_rows, llm_rows)):
        keys = ("left", "right", "expected_relation")
        if any(wordnet_row.get(key) != llm_row.get(key) for key in keys):
            raise ValueError(
                f"Classifier results do not match at benchmark row {index + 1}."
            )
        expected = wordnet_row["expected_relation"]
        wordnet_prediction = wordnet_row["predicted_relation"]
        llm_prediction = llm_row["predicted_relation"]
        if (
            expected not in RELATIONS
            or wordnet_prediction not in RELATIONS
            or llm_prediction not in RELATIONS
        ):
            raise ValueError("Classifier results contain an unsupported relation label.")
        paired_rows.append(
            {
                "left": wordnet_row["left"],
                "right": wordnet_row["right"],
                "expected_relation": expected,
                "wordnet_prediction": wordnet_prediction,
                "llm_prediction": llm_prediction,
                "wordnet_correct": wordnet_prediction == expected,
                "llm_correct": llm_prediction == expected,
            }
        )

    wordnet_correct = sum(bool(row["wordnet_correct"]) for row in paired_rows)
    llm_correct = sum(bool(row["llm_correct"]) for row in paired_rows)
    wordnet_only_correct = sum(
        bool(row["wordnet_correct"]) and not bool(row["llm_correct"])
        for row in paired_rows
    )
    llm_only_correct = sum(
        bool(row["llm_correct"]) and not bool(row["wordnet_correct"])
        for row in paired_rows
    )
    agreement_count = sum(
        row["wordnet_prediction"] == row["llm_prediction"] for row in paired_rows
    )
    wordnet_metrics = calculate_metrics(
        (row["expected_relation"] for row in paired_rows),
        (row["wordnet_prediction"] for row in paired_rows),
    )
    llm_metrics = calculate_metrics(
        (row["expected_relation"] for row in paired_rows),
        (row["llm_prediction"] for row in paired_rows),
    )

    per_class_comparison = []
    wordnet_per_class = wordnet_metrics["per_class"]
    llm_per_class = llm_metrics["per_class"]
    for label in sorted(set(wordnet_per_class) | set(llm_per_class)):
        wordnet_class = wordnet_per_class.get(label, {})
        llm_class = llm_per_class.get(label, {})
        per_class_comparison.append(
            {
                "relation": label,
                "support": wordnet_class.get("support", llm_class.get("support", 0)),
                "WordNet precision": wordnet_class.get("precision", 0.0),
                "LLM precision": llm_class.get("precision", 0.0),
                "WordNet recall": wordnet_class.get("recall", 0.0),
                "LLM recall": llm_class.get("recall", 0.0),
                "WordNet F1": wordnet_class.get("f1", 0.0),
                "LLM F1": llm_class.get("f1", 0.0),
            }
        )

    error_rows = [
        {
            "left": row["left"],
            "right": row["right"],
            "expected_relation": row["expected_relation"],
            "wordnet_prediction": row["wordnet_prediction"],
            "llm_prediction": row["llm_prediction"],
            "error_pattern": (
                "Both wrong"
                if not row["wordnet_correct"] and not row["llm_correct"]
                else "WordNet only wrong"
                if not row["wordnet_correct"]
                else "LLM only wrong"
            ),
        }
        for row in paired_rows
        if not row["wordnet_correct"] or not row["llm_correct"]
    ]
    shared_error_count = sum(
        not row["wordnet_correct"] and not row["llm_correct"]
        for row in paired_rows
    )
    wordnet_error_count = sum(
        not bool(row["wordnet_correct"]) for row in paired_rows
    )
    llm_error_count = sum(
        not bool(row["llm_correct"]) for row in paired_rows
    )

    wordnet_accuracy = float(wordnet_metrics["accuracy"])
    llm_accuracy = float(llm_metrics["accuracy"])
    wordnet_macro_f1 = float(wordnet_metrics["macro_f1"])
    llm_macro_f1 = float(llm_metrics["macro_f1"])
    if wordnet_accuracy == llm_accuracy:
        accuracy_conclusion = (
            f"Both classifiers have the same accuracy "
            f"({wordnet_correct}/{len(paired_rows)} correct)."
        )
    else:
        winner = "LLM" if llm_accuracy > wordnet_accuracy else "WordNet"
        accuracy_conclusion = (
            f"{winner} has higher accuracy on this benchmark "
            f"({max(wordnet_correct, llm_correct)}/{len(paired_rows)} correct)."
        )
    if wordnet_macro_f1 == llm_macro_f1:
        f1_conclusion = f"Macro-F1 is tied at {wordnet_macro_f1:.1%}."
    else:
        winner = "LLM" if llm_macro_f1 > wordnet_macro_f1 else "WordNet"
        winner_f1 = max(wordnet_macro_f1, llm_macro_f1)
        f1_conclusion = f"{winner} has higher macro-F1 ({winner_f1:.1%})."

    combined_recalls = [
        (
            (float(row["WordNet recall"]) + float(row["LLM recall"])) / 2,
            int(row["support"]),
            str(row["relation"]),
        )
        for row in per_class_comparison
    ]
    hardest_relation = min(combined_recalls, default=None)
    class_conclusion = (
        f"The lowest average per-class recall is for {hardest_relation[2]} "
        f"({hardest_relation[0]:.1%} across both classifiers; "
        f"support {hardest_relation[1]})."
        if hardest_relation is not None
        else "No per-class recall could be calculated."
    )
    error_pattern_counts = Counter(
        (
            str(row["expected_relation"]),
            str(row["wordnet_prediction"]),
            str(row["llm_prediction"]),
        )
        for row in error_rows
    )
    most_common_error = error_pattern_counts.most_common(1)
    error_conclusion = (
        "Most frequent error pattern: expected "
        f"{most_common_error[0][0][0]}, WordNet predicted "
        f"{most_common_error[0][0][1]}, and LLM predicted "
        f"{most_common_error[0][0][2]} "
        f"({most_common_error[0][1]} example(s))."
        if most_common_error
        else "Neither classifier made an error on this benchmark run."
    )

    conclusions = (
        accuracy_conclusion,
        f1_conclusion,
        (
            f"They agree on {agreement_count}/{len(paired_rows)} predictions "
            f"({agreement_count / len(paired_rows):.1%}). "
            f"On {wordnet_only_correct} rows only WordNet is correct; on "
            f"{llm_only_correct} rows only the LLM is correct."
        ),
        f"{len(error_rows)} rows contain at least one error; {shared_error_count} are shared errors.",
        class_conclusion,
        error_conclusion,
        (
            "These conclusions describe this small hand-labeled benchmark only; "
            "they do not establish statistical significance or general performance."
        ),
    )

    def confusion_matrix(prediction_key: str) -> list[dict[str, str | int]]:
        return [
            {
                "actual": actual,
                **{
                    predicted: sum(
                        row["expected_relation"] == actual
                        and row[prediction_key] == predicted
                        for row in paired_rows
                    )
                    for predicted in RELATIONS
                },
            }
            for actual in RELATIONS
        ]

    return {
        "count": len(paired_rows),
        "wordnet_metrics": wordnet_metrics,
        "llm_metrics": llm_metrics,
        "accuracy_delta": llm_accuracy - wordnet_accuracy,
        "macro_f1_delta": llm_macro_f1 - wordnet_macro_f1,
        "agreement_count": agreement_count,
        "agreement_rate": agreement_count / len(paired_rows),
        "wordnet_only_correct": wordnet_only_correct,
        "llm_only_correct": llm_only_correct,
        "wordnet_error_count": wordnet_error_count,
        "llm_error_count": llm_error_count,
        "shared_error_count": shared_error_count,
        "paired_rows": paired_rows,
        "error_rows": error_rows,
        "per_class_comparison": per_class_comparison,
        "wordnet_confusion": confusion_matrix("wordnet_prediction"),
        "llm_confusion": confusion_matrix("llm_prediction"),
        "conclusions": conclusions,
    }
