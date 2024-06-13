# 🧬 BioTalk 🗣️ : A Benchmark Dataset for Multimodal Prediction of Enzymatic Function Coupling DNA Sequences and Natural Language

Welcome to the BioTalk repository! 🎉 This dataset contains comprehensive data for predicting gene function from DNA sequences 🧬, accompanied by unstructured text descriptions 📝. Below, you'll find all the information you need to understand and use this dataset. 👇

## ⬇️ Download

You can download the dataset directly from following Link: [Dataset](https://drive.google.com/drive/folders/1lDpdfMCbW5MSgWoo7ZeAlAUFWkpbegYs)

The dataset is structured as follows:

```
.
├── Benchmark-Datasets-Train+Valid         
│    ├── Benchmark-I
│    │   ├── Train.parquet
│    │   └── Valid.parquet
│    ├── Benchmark-II
│    │   ├── Train.parquet
│    │   └── Valid.parquet
│    ├── Benchmark-III
│    │   ├── Train.parquet
│    │   └── Valid.parquet
│    └── Benchmark-IV
│        ├── Train.parquet
│        └── Valid.parquet
│
└── Benchmark-Datasets-Test
    ├── test1.csv
    └── test2.csv
    
```

### Benchmark Datasets

- **Benchmark-I:**
  - **Train.parquet:** Training data from UniProtKB/TrEMBL and UniProtKB/Swiss-Prot.
  - **Valid.parquet:** Validation data from the same combined dataset.
- **Benchmark-II:**
  - **Train.parquet:** Balanced training data based on EC number counts.
  - **Valid.parquet:** Balanced validation data.
- **Benchmark-III:**
  - **Train.parquet:** Training data exclusively from Swiss-Prot, with out-of-distribution entries removed.
  - **Valid.parquet:** Validation data from the same set.
- **Benchmark-IV:**
  - **Train.parquet:** Training data from Benchmark-III, balanced with a target of 10 examples per EC number.
  - **Valid.parquet:** Validation data similarly balanced.

### Test Datasets

- **test1.csv:** Test set derived from Benchmark-III as in-distribution test data.
- **test2.csv:** Test set derived from Benchmark-IV as balanced test data.

## 📊 Data Description

### Key Contributions:
1. **Novel Dataset:** Pairs DNA sequences with their functional descriptions, filling a critical gap in existing resources.
2. **Multimodal Applications:** Facilitates the development of multimodal models that predict DNA function in natural language.
3. **Unimodal and Multimodal Benchmarks:** Offers benchmarks for various models, including pretraining transformer models on DNA sequences.
4. **Impact:** Enhances the interpretability and utility of genomic data for a wide range of applications.

### Sample Data

Here's a preview of the dataset:


| AC      | EC         | OC      | UniRef90         | UniRef50         | EmblCdsId  | Sequence | UniRef100         | Description                                                                                                                  |
|---------|------------|---------|------------------|------------------|------------|----------|-------------------|------------------------------------------------------------------------------------------------------------------------------|
| F9UMS6  | 4.1.1.101  | Bacteria| UniRef90_F9UMS6  | UniRef50_F9UMS6  | CCC78515.1 | ATGACAAAAACTGCAAGTGA ... | UniRef100_F9UMS6  | The enzyme with the EC number 4.1.1.101 which is known as malolactic enzyme. It is ...  |
| A0A0A7GEY4 | 2.5.1.1  | Archaea | UniRef90_A0A0A7GEY4 | UniRef50_A0A0A7GEY4 | AIY90378.1 | ATGATTTCTGAGATAATTAA ... | UniRef100_A0A0A7GEY4  | Enzyme 2.5.1.1, identified as dimethylallyltranstransferase, is also known by geranyl-diphosphate synthase, prenyltransferase, ... |

### Load the Data

Here's an example of how to load the data using Python and pandas:

```python
import pandas as pd

# Load the training dataset
train_df = pd.read_parquet('Benchmark-Datasets-Train+Valid/Benchmark-I/Train.parquet')
valid_df = pd.read_parquet('Benchmark-Datasets-Train+Valid/Benchmark-I/Valid.parquet')

# Display the first few rows
print(train_df.head())
print(valid_df.head())
```

## 📏 Baselines

### Embeddings Quality Evaluation

This section provides Python scripts (`scores.py` and `k-nn_retrievals.py`) that utilize pre-trained transformer models to evaluate the quality of DNA sequence embeddings and to perform retrieval tasks for the top-k nearest embeddings.

**Repository Link:** [Embeddings Quality Evaluation](https://github.com/Hoarfrost-Lab/BioTalk/tree/main/Baselines/DNAembeddingQuality)

### EC Number Prediction Using DNA Embedding

We developed a two-layer classifier designed to predict Enzyme Commission (EC) numbers using various DNA embeddings, including DNABERT, Nucleotide Transformers, LOLBERT, and fine-tuned LOLBERT. This classifier was evaluated on two test datasets: test1 and test2.

The optimal hyperparameters were determined after a cross-validation process on the validation dataset, resulting in a configuration of a batch size of 64, a hidden size of 256, and a learning with a rate of 0.001. The model was trained for a duration of 10 epochs.

**Repository Link:** [EC Number Prediction](https://github.com/Hoarfrost-Lab/BioTalk/tree/main/Baselines/ECnumberPrediction)

### Multi-modal Zero- and Few-shot EC Number Predictions Using LLM Prompts

Our methodology exploits the multi-modal properties of our benchmark datasets by utilizing both DNA sequences and textual descriptions for EC number prediction. This is done using the open-access Llama 3 language model.

- **Zero-shot Prediction:** For zero-shot prompting, we provide natural language instructions that clearly describe the prediction task and outline the expected output. This strategy allows the LLM to construct a refined context that improves the accuracy of predictions. 

  **Code Repository:** [Zero-shot Prediction with Llama 3](https://github.com/Hoarfrost-Lab/BioTalk/blob/main/Baselines/MultimodalPrediction/llama3_zeroshot.ipynb)

- **Few-shot Learning:** In this approach, we select examples from our training data, simplify the instructions, and include three-shot examples to aid in model learning.

  **Code Repository:** [Few-shot Learning with Llama 3](https://github.com/Hoarfrost-Lab/BioTalk/blob/main/Baselines/MultimodalPrediction/llama3_fewshot.ipynb)
