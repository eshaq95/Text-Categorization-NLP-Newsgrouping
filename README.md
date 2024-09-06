# Text Categorization for 20 Newsgroups Dataset

## Project Overview
This project implements and compares multiple machine learning models for multi-class text classification using the 20 Newsgroups dataset. We explore traditional and deep learning approaches to categorize approximately 20,000 newsgroup documents into 20 different classes.

## Key Features
- Comprehensive data preprocessing and exploratory data analysis
- Implementation of four different models:
  - Multinomial Naive Bayes
  - Long Short-Term Memory (LSTM)
  - Convolutional Neural Network (CNN)
  - Bidirectional Encoder Representations from Transformers (BERT)
- Hyperparameter tuning for model optimization
- Comparative analysis of model performance

## Data
The project uses the 20 Newsgroups dataset, which consists of approximately 20,000 newsgroup documents across 20 different newsgroups. The dataset is obtained from the scikit-learn library.

## Methodology
1. Data Preprocessing:
   - Removal of headers, footers, quotes, email addresses, and URLs
   - Handling of contractions
   - Removal of non-alphabetic characters and lowercasing
   - Tokenization, stopword removal, and lemmatization

2. Exploratory Data Analysis:
   - Analysis of raw data
   - Visualization of document distribution across categories
   - Word cloud generation for common words
   - N-gram analysis

3. Model Implementation:
   - Multinomial Naive Bayes with TF-IDF vectorization
   - LSTM with word embeddings
   - CNN with word embeddings
   - Fine-tuned BERT model

4. Hyperparameter Tuning:
   - Grid search for Multinomial Naive Bayes, LSTM, and CNN
   - Optuna for BERT hyperparameter optimization

5. Evaluation:
   - Accuracy, Macro Average F1-Score, and Weighted Average F1-Score
   - Confusion matrix analysis

## Results
| Model               | Accuracy | Macro Avg F1-Score | Weighted Avg F1-Score |
|---------------------|----------|---------------------|------------------------|
| Multinomial NB      | 0.70     | 0.68                | 0.69                   |
| LSTM                | 0.64     | 0.63                | 0.64                   |
| CNN                 | 0.67     | 0.66                | 0.67                   |
| BERT                | 0.70     | 0.69                | 0.70                   |

BERT achieved the highest performance, closely followed by Multinomial Naive Bayes. CNN slightly outperformed LSTM, while all models struggled with closely related or overlapping topic classes.

## Usage
1. Clone the repository
2. Run the Jupyter notebooks in the `notebooks/` directory for step-by-step analysis and model implementation

## Future Work
- Further fine-tuning of models, especially BERT
- Exploration of data augmentation techniques
- Incorporation of domain-specific knowledge
- Investigation of ensemble methods

## Contributors
- Eshaq Rahmani
- Fellow Students

## Acknowledgments
- Scikit-learn for providing the 20 Newsgroups dataset
- The creators and maintainers of the libraries used in this project (TensorFlow, PyTorch, Transformers, etc.)
