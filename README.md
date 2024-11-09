# NLP Assignment: Sentiment Classification on Rotton Tomato Movie Reviews

  

**Course**: SC4002 / CE4045 / CZ4045 Natural Language Processing

**Academic Year**: 2024-2025

**Group ID**: G37

**Group Members**:

  

---

  

## Overview

This assignment implements a sentiment classification system using pretrained word embeddings, specifically GloVe, to classify movie reviews. The goal is to explore and compare various neural network architectures, including simple RNN, biLSTM, biGRU, CNN and Bert+LSTM models, and to assess the effectiveness of different methods for handling Out-of-Vocabulary (OOV) words.

  

## Files Included

-  `rnn.ipynb`: Notebook containing the implementation and training of the RNN model.

-  `bilstm.ipynb`: Notebook with the biLSTM model training and evaluation.

-  `bigru.ipynb`: Notebook with the biGRU model training and evaluation.

-  `cnn.ipynb`: Notebook with the cnn model training and evaluation.

-  `preprocessing.ipynb`: Notebook with the relevant pre-processing steps and the OOV words results.

  

## Instructions to Run the Code

  

### Dataset Setup

1. Ensure the required libraries are installed.

    !pip install -r requirements.txt

The libraries we used for this assignment are as follows: 

|Libraries|                                
|----------------|
|Numpy|
|Tensorflow/Keras|
|Scikit-Learn|
|Matplotlib|
|Deep Translator|
|Langdetect|
|spaCy|
|Contractions|
|NLTK|
|PySpellChecker|
|Random|
|Optuna|

2. Run the code to load the Rotten Tomatoes dataset.

3. The dataset will automatically be split into training, validation, and test sets.

  

### Running Each Model

1. Open the respective notebook (e.g., `rnn.ipynb` for RNN, `bilstm.ipynb` for biLSTM, `bigru.ipynb` for biGRU, `cnn.ipynb` for cnn, `bert+bilstm.ipynb` for bert+bilstm).

2. Follow the cells to load the dataset, preprocess data, and train the model.

3. After training, the notebook will display validation and test results for the model.

  

### Training and Evaluation Parameters

- For each model, the training parameters (e.g., batch size, learning rate) and validation accuracy per epoch are logged.

- Early stopping is applied based on validation accuracy to prevent overfitting.

  

### Running Enhancements (Part 3)

- Additional enhancements include updating embeddings during training, handling OOV words, and using alternative models

- These enhancements can be executed directly in the provided notebooks by following the respective cells.

  

## Sample Output

After training each model, accuracy scores on the test set are provided for comparison. For example:

  

-  **RNN with max pooling**: Accuracy on test set: 71.20%

-  **biLSTM**: Accuracy on test set: 81.71%

-  **biGRU**: Accuracy on test set: 79.26%

-  **CNN**: Accuracy on test set: 72.42%

-  **BERT+LSTM**: Accuracy on test set: 85.18%

  

Each notebook includes a summary of test accuracy, validation accuracy per epoch, and strategies used to handle OOV words.

  

## Additional Notes

-  **OOV Words**: We implemented a strategy to mitigate OOV word issues by using alternative word representations.

-  **Hyperparameter Tuning**: Each model’s hyperparameters were tuned based on validation set accuracy.

  

Please refer to the report for a complete analysis and explanation of model configurations, and results.
