import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
# Project 2: Bird Classification, Background Leakage, and HITL Cropping (CUB-200-2011)
**DA 351 (Advanced Descriptive Methods) — Spring 2026**  
**Author:** Khoi Van

## Introduction
This project extends my Homework 6 CNN workflow and focuses on a key interpretability concern: **background leakage**. In bird datasets, models can learn shortcuts from scenery (water, sky, trees) instead of learning bird morphology. That can produce apparently reasonable accuracy while failing the class philosophy of interpretable, human-centered modeling.

### Research Question
Does a human-in-the-loop (HITL) preprocessing strategy based on CUB-200-2011 bounding boxes improve model validity by reducing background leakage, compared with a baseline CNN trained on uncropped images?

### Ethical and Analytic Framing
- Shortcut learning can encode spurious correlations and mislead downstream users.
- We compare two pipelines with the same architecture/training plan to isolate the effect of human guidance.
- We use **Grad-CAM** as evidence of what the model attends to, not only how accurate it is.

### Dataset
We use the CUB-200-2011 metadata (`images.txt`, `image_class_labels.txt`, `train_test_split.txt`, `bounding_boxes.txt`) and focus on the same difficult pair from Homework 6:
- 132. White-crowned Sparrow
- 133. White-throated Sparrow
"""
    )
    return


@app.cell
def _():
    import marimo as mo
    import os
    from pathlib import Path
    import random

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    random.seed(351)
    np.random.seed(351)
    tf.random.set_seed(351)

    return (
        Path,
        accuracy_score,
        classification_report,
        confusion_matrix,
        mo,
        np,
        os,
        pd,
        plt,
        random,
        tf,
    )


@app.cell
def _(Path, pd):
    DATA_ROOT = Path("CUB_200_2011/CUB_200_2011")
    IMAGES_ROOT = DATA_ROOT / "images"

    images_df = pd.read_csv(
        DATA_ROOT / "images.txt",
        sep=" ",
        names=["image_id", "relative_path"],
    )
    labels_df = pd.read_csv(
        DATA_ROOT / "image_class_labels.txt",
        sep=" ",
        names=["image_id", "class_id"],
    )
    split_df = pd.read_csv(
        DATA_ROOT / "train_test_split.txt",
        sep=" ",
        names=["image_id", "is_train"],
    )
    bbox_df = pd.read_csv(
        DATA_ROOT / "bounding_boxes.txt",
        sep=" ",
        names=["image_id", "x", "y", "width", "height"],
    )
    classes_df = pd.read_csv(
        DATA_ROOT / "classes.txt",
        sep=" ",
        names=["class_id", "class_name"],
    )

    cub = (
        images_df.merge(labels_df, on="image_id")
        .merge(split_df, on="image_id")
        .merge(bbox_df, on="image_id")
        .merge(classes_df, on="class_id")
    )
    cub["filepath"] = cub["relative_path"].map(lambda p: str(IMAGES_ROOT / p))

    TARGET_CLASSES = {
        132: "132.White-crowned_Sparrow",
        133: "133.White-throated_Sparrow",
    }
    cub_pair = cub[cub["class_id"].isin(TARGET_CLASSES)].copy()

    cub_pair.head()
    return DATA_ROOT, TARGET_CLASSES, cub_pair


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Methods
### Experimental design
We fit two binary CNN pipelines with the same architecture:
1. **Baseline CNN**: trained on uncropped RGB images.
2. **HITL Crop CNN**: trained on images cropped to the human-annotated bird bounding box.

### Why this is appropriate
- Holding architecture/training settings constant isolates the causal role of preprocessing.
- Bounding boxes represent explicit human knowledge about where the bird is located.
- Grad-CAM helps evaluate whether each model's attention aligns with the scientific object (bird) versus context.

### Strengths and weaknesses
- Strength: clear A/B comparison tied to interpretability goals.
- Weakness: bounding boxes are expensive annotations and may still include some background.
- Weakness: binary task on one species pair limits broad generalization; this is a validity-focused extension.
"""
    )
    return


@app.cell
def _(Path, TARGET_CLASSES, cub_pair, np, os, tf):
    IMG_SIZE = (200, 200)
    BATCH_SIZE = 16
    EPOCHS = 5
    MAX_PER_CLASS = 120

    class_to_label = {132: 0, 133: 1}

    def decode_and_resize(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def decode_crop_and_resize(path, x, y, w, h):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        shape = tf.shape(img)
        ih = tf.cast(shape[0], tf.float32)
        iw = tf.cast(shape[1], tf.float32)

        x0 = tf.cast(tf.maximum(0.0, x), tf.int32)
        y0 = tf.cast(tf.maximum(0.0, y), tf.int32)
        x1 = tf.cast(tf.minimum(iw, x + w), tf.int32)
        y1 = tf.cast(tf.minimum(ih, y + h), tf.int32)

        crop_w = tf.maximum(1, x1 - x0)
        crop_h = tf.maximum(1, y1 - y0)

        img = tf.image.crop_to_bounding_box(img, y0, x0, crop_h, crop_w)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img

    sampled = []
    for class_id in TARGET_CLASSES:
        class_rows = cub_pair[cub_pair["class_id"] == class_id].sample(
            n=min(MAX_PER_CLASS, (cub_pair["class_id"] == class_id).sum()),
            random_state=351,
        )
        sampled.append(class_rows)
    pair_df = (
        tf.keras.utils.to_categorical([0]) and None
    )  # quick no-op to keep TF warm in marimo
    import pandas as _pd

    pair_df = _pd.concat(sampled).sort_values("image_id").reset_index(drop=True)
    pair_df["label"] = pair_df["class_id"].map(class_to_label)

    train_df = pair_df[pair_df["is_train"] == 1].copy()
    test_df = pair_df[pair_df["is_train"] == 0].copy()

    def build_baseline_dataset(df, training=False):
        ds = tf.data.Dataset.from_tensor_slices(
            (df["filepath"].values, df["label"].values.astype("int32"))
        )
        if training:
            ds = ds.shuffle(len(df), seed=351)
        ds = ds.map(lambda p, y: (decode_and_resize(p), y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def build_cropped_dataset(df, training=False):
        ds = tf.data.Dataset.from_tensor_slices(
            (
                df["filepath"].values,
                df["x"].values.astype("float32"),
                df["y"].values.astype("float32"),
                df["width"].values.astype("float32"),
                df["height"].values.astype("float32"),
                df["label"].values.astype("int32"),
            )
        )
        if training:
            ds = ds.shuffle(len(df), seed=351)
        ds = ds.map(
            lambda p, x, y, w, h, label: (decode_crop_and_resize(p, x, y, w, h), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_baseline = build_baseline_dataset(train_df, training=True)
    test_baseline = build_baseline_dataset(test_df)
    train_crop = build_cropped_dataset(train_df, training=True)
    test_crop = build_cropped_dataset(test_df)

    summary_df = pair_df.groupby(["class_id", "class_name", "is_train"]).size().reset_index(name="n")
    summary_df
    return (
        EPOCHS,
        IMG_SIZE,
        pair_df,
        test_baseline,
        test_crop,
        test_df,
        train_baseline,
        train_crop,
        train_df,
    )


@app.cell
def _(IMG_SIZE, tf):
    def make_cnn(input_shape=(200, 200, 3)):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="last_conv"),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
            metrics=["accuracy"],
        )
        return model

    baseline_model = make_cnn((*IMG_SIZE, 3))
    crop_model = make_cnn((*IMG_SIZE, 3))
    return baseline_model, crop_model


@app.cell
def _(EPOCHS, baseline_model, crop_model, test_baseline, test_crop, train_baseline, train_crop):
    history_baseline = baseline_model.fit(
        train_baseline,
        validation_data=test_baseline,
        epochs=EPOCHS,
        verbose=1,
    )
    history_crop = crop_model.fit(
        train_crop,
        validation_data=test_crop,
        epochs=EPOCHS,
        verbose=1,
    )
    return history_baseline, history_crop


@app.cell
def _(accuracy_score, baseline_model, classification_report, confusion_matrix, crop_model, np, pd, test_baseline, test_crop):
    def evaluate_binary(model, dataset, model_name):
        y_true = np.concatenate([y.numpy() for _, y in dataset], axis=0)
        y_prob = model.predict(dataset, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(
            y_true,
            y_pred,
            target_names=["White-crowned", "White-throated"],
            output_dict=True,
        )
        return {
            "model": model_name,
            "accuracy": acc,
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "precision_white_crowned": report["White-crowned"]["precision"],
            "recall_white_crowned": report["White-crowned"]["recall"],
            "precision_white_throated": report["White-throated"]["precision"],
            "recall_white_throated": report["White-throated"]["recall"],
        }

    baseline_metrics = evaluate_binary(baseline_model, test_baseline, "Baseline (Uncropped)")
    crop_metrics = evaluate_binary(crop_model, test_crop, "HITL Bounding Box Crop")

    metrics_df = pd.DataFrame([baseline_metrics, crop_metrics])
    metrics_df
    return metrics_df


@app.cell
def _(history_baseline, history_crop, metrics_df, mo, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_baseline.history["accuracy"], label="Baseline Train")
    axes[0].plot(history_baseline.history["val_accuracy"], label="Baseline Val")
    axes[0].plot(history_crop.history["accuracy"], label="Crop Train")
    axes[0].plot(history_crop.history["val_accuracy"], label="Crop Val")
    axes[0].set_title("Accuracy by Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].bar(metrics_df["model"], metrics_df["accuracy"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Holdout Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].tick_params(axis="x", rotation=12)

    plt.tight_layout()
    mo.as_html(fig)
    return


@app.cell
def _(IMG_SIZE, baseline_model, crop_model, np, plt, test_df, tf):
    def gradcam_heatmap(model, image_tensor, layer_name="last_conv"):
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(image_tensor)
            loss = preds[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    sample_row = test_df.iloc[0]
    raw = tf.image.decode_jpeg(tf.io.read_file(sample_row["filepath"]), channels=3)
    raw_f = tf.cast(raw, tf.float32) / 255.0

    baseline_in = tf.image.resize(raw_f, IMG_SIZE)[None, ...]

    x0 = int(max(0, sample_row["x"]))
    y0 = int(max(0, sample_row["y"]))
    x1 = int(min(raw.shape[1], sample_row["x"] + sample_row["width"]))
    y1 = int(min(raw.shape[0], sample_row["y"] + sample_row["height"]))
    cropped = raw_f[y0:y1, x0:x1, :]
    crop_in = tf.image.resize(cropped, IMG_SIZE)[None, ...]

    hb = gradcam_heatmap(baseline_model, baseline_in)
    hc = gradcam_heatmap(crop_model, crop_in)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes[0, 0].imshow(baseline_in[0])
    axes[0, 0].set_title("Baseline Input")
    axes[0, 1].imshow(baseline_in[0])
    axes[0, 1].imshow(hb, cmap="jet", alpha=0.45)
    axes[0, 1].set_title("Baseline Grad-CAM")

    axes[1, 0].imshow(crop_in[0])
    axes[1, 0].set_title("Cropped Input (HITL)")
    axes[1, 1].imshow(crop_in[0])
    axes[1, 1].imshow(hc, cmap="jet", alpha=0.45)
    axes[1, 1].set_title("Cropped Grad-CAM")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Results
- The table reports holdout accuracy, confusion-matrix counts, and class-level precision/recall for each model.
- The learning-curve panel compares convergence and overfitting behavior.
- The Grad-CAM panel provides qualitative attention evidence.

### What to look for
- If the HITL crop model improves accuracy and/or recall balance, that supports reduced shortcut dependence.
- In Grad-CAM, the strongest evidence is heat concentrated on the bird body/head for cropped images.
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Interpretation
This project prioritizes **interpretability and validity** over raw score maximization.

If the cropped model attends more consistently to the bird and performs at least comparably, then human guidance is improving model behavior in a way that aligns with DA 351 course goals. Even if accuracy gains are modest, a shift in attention maps away from background context is an important scientific result: the model is learning a more causally plausible signal.

### Caveats and limits
- Binary species design limits generalization to all 200 classes.
- Bounding boxes may still include habitat cues.
- Grad-CAM is suggestive, not a full causal proof of feature use.

### Future extension
A direct next step is to expand to multiple class pairs and compare three settings: uncropped, bounding-box cropped, and segmentation-mask cropped, then quantify overlap between Grad-CAM heatmaps and annotated bird regions.
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## Uses of Python (Technical Reflection)

| Dependency | Role in project | Rationale |
|---|---|---|
| marimo | Notebook workflow + markdown narrative | Reproducible, literate analysis in one file |
| pandas | Metadata joins and experiment table prep | Reliable tabular preprocessing |
| tensorflow/keras | CNN modeling, tf.data pipeline, Grad-CAM gradients | Efficient deep learning stack already used in HW6 |
| matplotlib | Training curves and Grad-CAM visualizations | Clear communication for non-technical readers |
| scikit-learn | Accuracy, confusion matrix, precision/recall | Standard classification diagnostics |

### Reproducibility and readability choices
- Fixed random seeds for numpy/tensorflow.
- Shared architecture and optimizer between variants.
- Helper functions for dataset construction and model evaluation to avoid copy/paste drift.
"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
## References
- Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S. (2011). *The Caltech-UCSD Birds-200-2011 Dataset*. California Institute of Technology.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *Proceedings of ICCV*.
- Chollet, F. et al. (TensorFlow/Keras documentation).
"""
    )
    return


if __name__ == "__main__":
    app.run()
