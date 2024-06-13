# 🧬 BioTalk 🗣️ : A Benchmark Dataset for Multimodal Prediction of Enzymatic Function Coupling DNA Sequences and Natural Language

Welcome to the **BioTalk** repository! This dataset contains comprehensive data for predicting gene function from DNA sequences, accompanied by unstructured text descriptions. Below, you'll find all the information you need to understand and use this dataset.

### Download

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