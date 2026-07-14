from __future__ import annotations

from typing import Any


def build_cross_model_rows(
    predictions_by_model: dict[str, list[dict[str, Any]]],
    teacher_student_pairs: dict[str, tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge aligned per-model predictions and mark cross-model error patterns."""
    if not predictions_by_model:
        return [], []

    model_keys = list(predictions_by_model)
    indexed = {
        model: {int(row["test_index"]): row for row in rows}
        for model, rows in predictions_by_model.items()
    }
    expected_indices = set(indexed[model_keys[0]])
    for model, rows_by_index in indexed.items():
        if set(rows_by_index) != expected_indices:
            raise ValueError(f"Prediction indices for {model} do not match the other models.")

    combined_rows = []
    transition_rows = []
    for index in sorted(expected_indices):
        first = indexed[model_keys[0]][index]
        row: dict[str, Any] = {
            "test_index": index,
            "image_id": first["image_id"],
            "source_path": first["source_path"],
            "source_record_index": first["source_record_index"],
            "true_class": first["true_class"],
        }
        failed_models = []
        for model in model_keys:
            prediction = indexed[model][index]
            if prediction["true_class"] != first["true_class"]:
                raise ValueError(f"True label mismatch for test index {index}.")
            correct = bool(prediction["top1_correct"])
            row[f"{model}_predicted_class"] = prediction["predicted_class"]
            row[f"{model}_confidence"] = prediction["confidence"]
            row[f"{model}_correct"] = correct
            row[f"{model}_failure_image"] = prediction["saved_image"]
            if not correct:
                failed_models.append(model)

        row["failure_count"] = len(failed_models)
        row["failed_models"] = ";".join(failed_models)
        row["failed_three_or_more"] = len(failed_models) >= 3
        row["failed_all_models"] = len(failed_models) == len(model_keys)

        has_transition = False
        for pair_name, (teacher, student) in teacher_student_pairs.items():
            if teacher not in indexed or student not in indexed:
                continue
            teacher_correct = bool(indexed[teacher][index]["top1_correct"])
            student_correct = bool(indexed[student][index]["top1_correct"])
            recovery = not teacher_correct and student_correct
            regression = teacher_correct and not student_correct
            row[f"{pair_name}_teacher_failed_student_passed"] = recovery
            row[f"{pair_name}_student_failed_teacher_passed"] = regression
            has_transition = has_transition or recovery or regression

        combined_rows.append(row)
        if has_transition:
            transition_rows.append(dict(row))

    return combined_rows, transition_rows
