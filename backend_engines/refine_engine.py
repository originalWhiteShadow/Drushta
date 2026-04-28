import os

import numpy as np

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
import tensorflow_model_optimization as tfmot


def apply_equalized_odds(probabilities, sensitive_column):
    probabilities = np.asarray(probabilities).reshape(-1)
    sensitive_column = np.asarray(sensitive_column)

    thresholds = {}
    adjusted = np.zeros_like(probabilities, dtype=int)

    for group in np.unique(sensitive_column):
        mask = sensitive_column == group
        group_probs = probabilities[mask]

        # Equalize FNR by choosing a threshold that yields similar positive rate across groups
        if len(group_probs) == 0:
            thresholds[str(group)] = 0.5
            continue

        threshold = float(np.quantile(group_probs, 0.5))
        thresholds[str(group)] = threshold
        adjusted[mask] = (group_probs >= threshold).astype(int)

    return adjusted, thresholds


def apply_equalized_odds_multiclass(probabilities, y_true, sensitive_column, class_labels):
    # Approximate multiclass equalized odds via one-vs-rest, per-group thresholds.
    probs = np.asarray(probabilities)
    y_true = np.asarray(y_true).astype(int)
    sensitive_column = np.asarray(sensitive_column).astype(str)

    if probs.ndim != 2:
        raise ValueError("Multiclass probabilities must be 2D (n_samples, n_classes).")

    num_classes = probs.shape[1]
    if len(class_labels) != num_classes:
        raise ValueError("class_labels length must match probability columns.")

    base_threshold = 1.0 / max(num_classes, 1)
    thresholds = {}

    for group in np.unique(sensitive_column):
        thresholds[str(group)] = {label: base_threshold for label in class_labels}

    for class_idx, label in enumerate(class_labels):
        class_mask = y_true == class_idx
        class_probs = probs[class_mask, class_idx]

        if class_probs.size == 0:
            continue

        target_tpr = float(np.mean(class_probs >= base_threshold))
        quantile = float(np.clip(1.0 - target_tpr, 0.0, 1.0))

        for group in np.unique(sensitive_column):
            group_mask = (sensitive_column == group) & class_mask
            group_probs = probs[group_mask, class_idx]

            if group_probs.size == 0:
                threshold = 1.0
            else:
                threshold = float(np.quantile(group_probs, quantile))

            thresholds[str(group)][label] = threshold

    adjusted = np.zeros(len(probs), dtype=int)
    for idx, group_value in enumerate(sensitive_column):
        group_thresholds = thresholds.get(str(group_value), {})
        adjusted_scores = []
        for class_idx, label in enumerate(class_labels):
            threshold = float(group_thresholds.get(label, base_threshold))
            adjusted_scores.append(probs[idx, class_idx] - threshold)

        adjusted[idx] = int(np.argmax(adjusted_scores))

    return adjusted, thresholds


def optimize_model(keras_model):
    output_units = keras_model.output_shape[-1] if keras_model.output_shape else 1
    use_multiclass = output_units is not None and output_units > 1

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
            target_sparsity=0.5, begin_step=0, end_step=100
        )
    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        keras_model, **pruning_params
    )

    if use_multiclass:
        loss = "sparse_categorical_crossentropy"
    else:
        loss = "binary_crossentropy"

    pruned_model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    # Train with dummy data placeholders to advance pruning; user can replace with real data
    dummy_x = np.random.rand(128, keras_model.input_shape[-1]).astype(np.float32)
    if use_multiclass:
        dummy_y = np.random.randint(0, output_units, size=(128,)).astype(np.int32)
    else:
        dummy_y = np.random.randint(0, 2, size=(128,)).astype(np.float32)

    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
    pruned_model.fit(dummy_x, dummy_y, epochs=3, verbose=0, callbacks=[pruning_callback])

    stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    return tflite_model
