# EC Number Prediction using DNA Embeddings

This section is dedicated to predicting Enzyme Commission (EC) numbers using a two-layer classifier and various DNA embeddings, including DNABERT, Nucleotide Transformers, LOLBERT, and fine-tuned LOLBERT.

## Project Structure

The folder includes the following key files:

- `environment.yml`: Contains all the dependencies required to run the project.
- `embedding.py`: Script for generating DNA embeddings using pre-trained models.
- `dataset.py`: Handles data loading and preprocessing.
- `model.py`: Defines the two-layer classifier architecture.
- `train.py`: Script for training the classifier.
- `test.py`: Script for evaluating the classifier on the test dataset.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository:**

2. **Create and Activate the Environment:**

```python
conda env create -f environment.yml
conda activate dna_env
```
3. **Run the Training Script:**

To train the model, run the following command, adjusting the parameters as needed:

```python
python train.py \
--train_csv_path PATH_TO_TRAIN_CSV \
--embedding_type EMBEDDING_TYPE \
--num_epochs 25 \
--batch_size 64 \
--learning_rate 0.001 \
--hidden_size 256 \
--model_save_path trained_model.pth \
--metrics_save_path training_metrics.json \
--save_embeddings
```
- `--train_csv_path`: Specify the path to your training CSV file.
- `--embedding_type`: Choose the type of embedding (`dnabert`, `nutransformer`, `lolbert4`).
- `--num_epochs`: Set the number of epochs for training (default is 25).
- `--batch_size`: Set the batch size for training (default is 64).
- `--learning_rate`: Set the learning rate for the optimizer (default is 0.001).
- `--hidden_size`: Set the hidden layer size (default is 256).
- `--model_save_path`: Specify where to save the trained model (default is `trained_model.pth`).
- `--metrics_save_path`: Specify where to save training metrics (default is `training_metrics.json`).
- `--save_embeddings`: Include this flag if you want to save the embeddings.

4. **Evaluate the Model:**

To evaluate the model, use the following command with appropriate parameters:

```python
python test.py \
--test_csv_path PATH_TO_TEST_CSV \
--embedding_type EMBEDDING_TYPE \
--model_path MODEL_PATH \
--num_classes 1196 \
--save_embeddings \
--metrics_save_path test_metrics.json \
--predictions_save_path updated_test.csv
```

- `--test_csv_path`: Specify the path to your test CSV file.
- `--embedding_type`: Choose the type of embedding (`dnabert`, `nutransformer`, `lolbert4`).
- `--model_path`: Specify the path to the trained model.
- `--num_classes`: Number of classes for the test (default is 1196).
- `--save_embeddings`: Include this flag if you want to save the embeddings during testing.
- `--metrics_save_path`: Specify where to save test metrics (default is `test_metrics.json`).
- `--predictions_save_path`: Specify where to save the updated test CSV (default is `updated_test.csv`).
