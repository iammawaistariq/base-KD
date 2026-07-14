from vim_kd.audit import build_cross_model_rows


def prediction(index, true_class, predicted_class):
    return {
        "test_index": index,
        "image_id": f"image_{index}",
        "source_path": "/data/test_batch",
        "source_record_index": index,
        "true_class": true_class,
        "predicted_class": predicted_class,
        "confidence": 0.9,
        "top1_correct": true_class == predicted_class,
        "saved_image": "" if true_class == predicted_class else f"/failures/{index}.png",
    }


def test_cross_model_failure_and_teacher_student_markers():
    predictions = {
        "teacher": [
            prediction(0, "cat", "dog"),
            prediction(1, "ship", "ship"),
        ],
        "student": [
            prediction(0, "cat", "cat"),
            prediction(1, "ship", "truck"),
        ],
        "third": [
            prediction(0, "cat", "dog"),
            prediction(1, "ship", "truck"),
        ],
    }

    rows, transitions = build_cross_model_rows(
        predictions,
        {"pair": ("teacher", "student")},
    )

    assert rows[0]["pair_teacher_failed_student_passed"] is True
    assert rows[0]["pair_student_failed_teacher_passed"] is False
    assert rows[1]["pair_student_failed_teacher_passed"] is True
    assert rows[1]["failure_count"] == 2
    assert rows[1]["failed_models"] == "student;third"
    assert len(transitions) == 2


def test_marks_failures_by_three_and_all_selected_models():
    failed = prediction(0, "bird", "cat")
    predictions = {name: [dict(failed)] for name in ("one", "two", "three")}

    rows, _ = build_cross_model_rows(predictions, {})

    assert rows[0]["failed_three_or_more"] is True
    assert rows[0]["failed_all_models"] is True
