# Newsgroup Text Classification

This project aims to classify newsgroup documents into one of the 20 newsgroup categories using 4 models.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [How to run the code](#how-to-run-the-code)

## Introduction

The problem being solved in this project is the classification of newsgroup documents into one of the 20 categories. The model is trained using the Keras library and pre-trained GloVe embeddings. The code includes sections for data preparation, model training and evaluation, hyperparameter optimization, and confusion matrix visualization.

## Requirements

- Python 3.x
- Keras
- TensorFlow
- Scikit-learn
- Scikit-learn Extras (scikeras)
- NumPy
- Matplotlib
- Seaborn
- Contractions
- OpenAI
- Transformers
- Wordcloud
- Optuna

The installation of these libraries is included in the '1.1 Installing Libraries and Importing Dependencies' section of the notebook file.

## How to run the code

The code for this project is contained in a single Jupyter Notebook file, which is divided into two main sections indicated by bold headings:

1. **Dataset Exploration, Descriptive Analysis, and Preprocessing**
2. **Model Training and Evaluation**

The code is structured in such a way that it should be executed in the order it appears, with each cell relying on the outputs of previous cells.

We have employed intermediate pipelines for preprocessing, vectorization, and tokenization/padding. The outputs of each pipeline are fed into the subsequent pipelines to ensure consistency and prevent data leakage. Detailed explanations of these pipelines can be found within the notebook.

To run the Multinomial model, simply ensure that the preprocessing and vectorization pipelines have been executed first. For the CNN and LSTM models, you will need to run the tokenizer and padder intermediate pipeline before running the models. Additionally, please ensure that the appropriate GloVe word embeddings are installed:

- Wikipedia 2014 + Gigaword 5 (6B tokens with 100d, 200d, and 300d vectors)
- Common Crawl (840B tokens with 300d vectors)

Make sure to set the correct file paths to the text files in the model implementations. With these considerations, running the notebook from top to bottom should result in smooth execution without any additional adjustments needed.
