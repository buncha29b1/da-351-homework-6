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

    return Counter, mo, pd, tf


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


@app.cell
def _():
    # your code here 
    return


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
