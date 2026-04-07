import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    __[Khoi Van]__

    # Coding Homework 6: Advanced Image Classification

    ## Assignment Setup

    This assignment assumes you have created a new virtual enviroment with Python 3.12 and installed `marimo`, `pandas`, `seaborn`, `opencv`, `sklearn`, and `tensorflow` in this new environment.

    Run the code cells below to confirm. If the final code cell before section II, "Coding Homework," prints an accuracy score, everything is working properly.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import tensorflow as tf
    import pandas as pd
    from collections import Counter
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    return (
        Counter,
        accuracy_score,
        classification_report,
        confusion_matrix,
        mo,
        np,
        os,
        pd,
        plt,
        tf,
    )


@app.cell
def _(tf):
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 200x200 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.25),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 
        tf.keras.layers.Dense(1, activation='sigmoid')])

    model.summary()
    return (model,)


@app.cell
def _(model, tf):
    RMSprop = tf.keras.optimizers.RMSprop
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    return


@app.cell
def _(tf):
    training_set = tf.keras.utils.image_dataset_from_directory(
      'sample_birds',
      seed=123,
      image_size=(200, 200),
      subset='training',
      validation_split=0.3,
      batch_size=5)

    validation_set = tf.keras.utils.image_dataset_from_directory(
      'sample_birds',
      shuffle=True,
      seed=17,
      image_size=(200, 200),
      validation_split=0.3,
      subset='validation',
      batch_size=5)

    holdout_set_all = tf.keras.utils.image_dataset_from_directory(
      'sample_birds',
      shuffle=False,
      seed=17,
      image_size=(200, 200),
      batch_size=1) # batch size has to be one for this set

    train_file_paths = training_set.file_paths
    validation_file_paths = validation_set.file_paths
    holdout_file_paths = holdout_set_all.file_paths
    return (
        holdout_file_paths,
        holdout_set_all,
        training_set,
        validation_file_paths,
        validation_set,
    )


@app.cell
def _(holdout_file_paths, holdout_set_all, validation_file_paths):
    images = []
    labels = []
    for e, image_label in enumerate(holdout_set_all):
        f = holdout_file_paths[e]
        if f in validation_file_paths:
            images.append(image_label[0].numpy())
            labels.append(image_label[1].numpy())
    len(validation_file_paths), len(images), len(labels)
    return images, labels


@app.cell
def _(model, training_set, validation_set):
    history = model.fit(training_set,
          epochs=11,
          verbose=1,
          validation_data = validation_set)

    model.evaluate(validation_set)
    return


@app.cell
def _(images, model, tf):
    img_arrays = []
    for i in images:
        img_arrays.append(i)

    test_dataset = tf.data.Dataset.from_tensor_slices(img_arrays)
    preds = model.predict(test_dataset)
    preds[0]
    return (preds,)


@app.cell
def _(Counter, labels):
    c = Counter([i[0] for i in labels])
    c
    return (c,)


@app.cell
def _(c, labels, pd, preds):
    df = pd.DataFrame()
    df['true_label'] = [i[0] for i in labels]
    df['predict_probability'] = [i[0] for i in preds]
    df = df.sort_values(by='predict_probability')
    inferred_labels = [0 for i in range(c[0])] + [1 for i in range(c[1])]
    df['predicted_label'] = inferred_labels 
    df['correct'] = df['true_label'] == df['predicted_label']
    df
    return (df,)


@app.cell
def _(df):
    len(df.loc[df['correct'] == True])/len(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## II. Coding Homework

    To complete this assignment, you will need to:

    1. Download the full dataset and unzip the compressed files
    2. Select bird species to classify (see below)
    3. Set up files and folders for classification
    4. Adapt the code above to train and evaluate various classifiers



    ### 1. Download the full dataset and unzip the compressed files

    Dataset link is located in `README.md`. The CUB 200 dataset has a unique id number and a species name in each folder name.

    ### 2. Select Bird Species to Classify

    Identify three sets of bird pairings and write a hypothesis about how difficult it will be for a model to differentiate the species from one another. You should select pairings so that you can reasonably predict different levels of success (e.g. an easy, medium, and hard task). For example, the sample pairing is the Black-footed Albatross vs. Artic Tern, and we might predict this classification task to be relatively easy.  (When making your selections, please do not use the sample pairing.)

    ### 3. Set up Files and Folders

    Tensorflow's `image_dataset_from_directory` method is _much_ faster than openCV, but it requires your files and folders to be set up in a specific way. You will need move files around as you go to create the following structure:

    ```
    > parent_folder
    	> class_a_folder
    		class_a_img_1
    		class_a_img_2
    		etc.
    	> class_b_folder
    		class_b_img_1
    		class_b_img_2
    		etc.
    ```

    Once you have this structure, you can use it to define training and validation sets. Labels can be supplied, but it's much easier to have Tensorflow infer them from the folder names.

    __Note:__ I have set up the sample_birds folder this way as a guide for you. You can easily create three new folders and call them `easy`, `medium`, and `hard` respectively.

    ### 4. Train and Evaluate Your Models

    Once your setup is complete, you will train and validate binary classification models for all of your pairs. Every model should be a CNN with the same architecture. (You can modify the architecture I've provided, but you shouldn't use different setups for different classifiers.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Select Bird Species to Classify

    #### Easy Task: 100. Brown Pelican vs. 068. Ruby-throated Hummingbird

    - Hypothesis: Differentiating these two species will be an easy task for the model. These birds have drastically distinct morphologies, sizes, and characteristic environments. The pelican features a large body, distinctive bill, and a predominantly brown/gray color palette, whereas the hummingbird is tiny with iridescent green and red feathers. The Convolutional Neural Network (CNN) should rapidly learn these obvious macroscopic and color-based differences, resulting in high accuracy and few misclassifications.

    #### Medium Task: 073. Blue Jay vs. 017. Cardinal

    - Hypothesis: This task will present a medium level of difficulty. Both species are similarly sized passerine birds with crests on their heads, and they are often photographed in similar environments (such as backyard feeders or tree branches). While their physical silhouettes and backgrounds share similarities, they have stark, contrasting primary colors (bright blue vs. bright red). The model will likely rely heavily on these color channel differences to achieve a reasonably high accuracy, but may make occasional errors if lighting conditions obscure the colors or if the images are predominantly in grayscale.

    #### Hard Task: 132. White-crowned Sparrow vs. 133. White-throated Sparrow

    - Hypothesis: Differentiating these two sparrows will be a hard task. Because they belong to the same family, they share nearly identical body shapes, beak structures, typical postures, and overall brown-streaked plumage. The primary visual differences are fine-grained, localized details on their heads (the presence of a white throat patch vs. specific crown striping). A basic CNN might struggle to focus on these localized, subtle features without an attention mechanism, leading to a higher rate of false positives and false negatives, and an overall lower accuracy.
    """)
    return


@app.cell
def _(model, np, os, tf):
    # Define the parent folder paths based on your task difficulty pairings
    # Ensure these match the actual names of your parent folders
    parent_folders = ['easy', 'medium', 'hard']

    # Standard image size and batch size for the model
    IMG_SIZE = (200, 200) 
    BATCH_SIZE = 32

    def evaluate_task(parent_folder):
        print(f"==================================================")
        print(f"RESULTS FOR: {parent_folder.upper()} TASK")
        print(f"==================================================")

        # 1. Load data using the specified folder structure
        # image_dataset_from_directory automatically treats subfolders (class_a_folder, class_b_folder) as classes
        if not os.path.exists(parent_folder):
            print(f"Directory '{parent_folder}' not found. Please ensure the path is correct.\n")
            return

        dataset = tf.keras.utils.image_dataset_from_directory(
            parent_folder,
            shuffle=False, # Shuffle must be False to align predictions with true labels
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        class_names = dataset.class_names
        print(f"Comparing: {class_names[0]} vs {class_names[1]}\n")

        # Extract true labels sequentially from the dataset
        y_true = np.concatenate([y for x, y in dataset], axis=0)

        # Assuming your trained model is stored in a variable named 'model'
        # If you have separate models for each task, you would load them dynamically here 
        # e.g., model = tf.keras.models.load_model(f'{parent_folder}_model.keras')

        try:
            # Generate predictions
            y_pred_probs = model.predict(dataset)

            # Convert probabilities to binary class predictions (0 or 1)
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()

        except NameError:
            print("Error: The 'model' variable is not defined. Make sure you have trained and assigned your model.")

    # Run the evaluation loop over the 3 folder structures
    for folder in parent_folders:
        evaluate_task(folder)
    return BATCH_SIZE, IMG_SIZE, folder, parent_folders


@app.cell
def _(mo):
    mo.md(r"""
    ### Results Code

    For each model, report the following results:

    1. Overall accuracy of the model
    2. A confusion matrix of your validation set's True Positives, True Negatives, False Negatives, and False Positives
    3. Per class precision and recall (using `scikit-learn` functions)
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    IMG_SIZE,
    accuracy_score,
    classification_report,
    confusion_matrix,
    folder,
    model,
    np,
    os,
    parent_folders,
    plt,
    tf,
):
    for folder1 in parent_folders:
        print(f"==================================================")
        print(f"METRICS FOR: {folder1.upper()} TASK")
        print(f"==================================================")

        if not os.path.exists(folder1):
            print(f"Directory '{folder1}' not found. Skipping...\n")
            continue

        dataset = tf.keras.utils.image_dataset_from_directory(
            folder1,
            shuffle=False,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        # Extract true labels
        y_true = np.concatenate([y for x, y in dataset], axis=0)

        # Generate predictions
        y_pred_probs = model.predict(dataset)
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()

        # 1. Report Overall accuracy
        acc = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy: {acc:.4f}\n")

        # 2. Report Confusion Matrix using matplotlib
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
        plt.title(f"Confusion Matrix: {folder.capitalize()} Task", pad=20)
        fig.colorbar(cax)

        # Annotate TP, TN, FP, FN counts directly on the matrix
        for k in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, k, str(cm[k, j]), va='center', ha='center', fontsize=12, fontweight='bold')

        plt.xlabel('Predicted Label', fontsize=11)
        plt.ylabel('True Label', fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(dataset.class_names)
        ax.set_yticklabels(dataset.class_names)
        plt.show()

        # 3. Report Per class precision and recall
        print("Classification Report (Precision & Recall):")
        print(classification_report(y_true, y_pred, target_names=dataset.class_names))
        print("\n")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interpretation

    1. Interpret your output, highlighting the key results and explaining the main takeaways. Revisit your hypotheses from the introduction. How did your models do compared to your hypotheses? What was surprising and why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [ your written response here]
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    2. Discuss the precision and recall scores of each class for your various models. What seems to be your strongest model and why?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [ your written response here]
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    3. As a whole, what does this analysis tell us? What are the strengths/limitations of  this data set? What are the strengths/limitations of this method? What is one future direction you could envision for future data analysts or data collectors?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    4. Take a step back and analyze your own use of code. Provide some rationale for choices you've made. How did you (or how might we) refactor the code to avoid repeating the same blocks three times?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [ your written response here]
    """)
    return


if __name__ == "__main__":
    app.run()
