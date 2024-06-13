# DNA Sequence Embedding Quality and Top-K NN Retrieval with Transformers

This repository contains Python scripts (`scores.py` and `k-nn_retrievals.py`) that utilize pre-trained transformer models for evaluating DNA sequence embedding quality and retrieval of top k nearest embeddings tasks.

## Dependencies

The scripts rely on the following external libraries:

- `torch`
- `transformers`
- `sklearn`
- `numpy`
- `pandas`
- `faiss`

These libraries can be installed using pip:

```
pip install torch transformers sklearn pandas numpy faiss
```

## Pre-trained Models and Tokenizers

The scripts assume you have pre-trained transformer models and their corresponding tokenizers available locally. The model and tokenizer names are specified in the scripts themselves (e.g., `models/LOLBERTv4/`). Make sure to download and place the appropriate models and tokenizers in the designated directories.

## Functionality

The scripts utilize helper functions defined in a separate file utils/utils.py. These functions handle common tasks like data loading, tokenizer interaction, and embedding retrieval.

The scripts leverage a custom tokenizer class named BioTokenizer. This tokenizer is specifically designed for processing DNA sequences and while overrides the tokenization method of the base RobertaTokenizer. 
The scripts perform the following tasks:

### scores.py:

- Loads protein/DNA sequences and Enzyme Commission (EC) labels from a DataFrame.
- Selects a subset of sequences for evaluation.
- Loads a pre-trained transformer model (currently `RobertaModel`) and tokenizer from the specified directories.
- Extracts various embedding statistics for each sequence (class token, minimum, maximum, average).
- Calculates silhouette score for each embedding statistic to evaluate class separation.

### k-nn_retrievals.py:

- Loads protein/DNA sequences and EC labels for training and testing sets from DataFrames.
- Loads a pre-trained transformer model (currently `RobertaModel`) and tokenizer.
- Extracts embeddings for both training and testing sequences.
- Calculates various embedding statistics for each sequence (same as `scores.py`).
- Performs k-Nearest Neighbors (KNN) retrieval for each test sequence based on the chosen embedding statistic and different values of K (number of neighbors).
- Evaluates the retrieval accuracy by comparing predicted EC labels with actual labels for the nearest neighbors.

## Running the Scripts

1. Make sure you have the required libraries and data files available.
2. Update the paths to your pre-trained models and tokenizers in the scripts if necessary.
3. Navigate to the directory containing the scripts and run:

```
python scores.py
python k-nn_retrievals.py
```




These commands will execute the scripts and print the evaluation results (silhouette scores for scores.py and retrieval accuracy for k-nn_retrievals.py).
