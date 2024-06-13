import numpy as np
import pandas as pd
import torch
import os, csv, sys, glob
from scipy.spatial.distance import jaccard, cosine, euclidean, sqeuclidean
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
# from sklearn.metrics import silhouette_score
from sklearn import metrics
from collections import Counter
import faiss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BioTokenizer import BioTokenizer


def overlap_score(neighbor_ec: str, true_ec: str) -> tuple[int, int, int]:
    """Calculates the overlap score between two Enzyme Commission (EC) numbers.

    The overlap score is the number of levels (separated by periods) that match exactly
    between the two EC numbers. This function can be useful for comparing the specificity
    of EC numbers in enzyme classification.

    Args:
        neighbor_ec (str): The first EC number (e.g., "1.2.3.4").
        true_ec (str): The second EC number (e.g., "1.2.3.5").

    Returns:
        tuple[int, int, int]: A tuple containing:
            - overlap_score (int): The number of matching levels from the beginning.
            - alpha (int): The length (number of levels) of the first EC number.
            - beta (int): The length (number of levels) of the second EC number.

    Examples:
        >>> overlap_score("1.2.3.4", "1.2.3.5")
        (3, 4, 4)

         >>> overlap_score("1", "1.2")
        (1, 1, 2)

        >>> overlap_score("invalid_ec", "1.2.3")  # Raises ValueError
        ValueError: Invalid EC number format.
    """
    if not isinstance(neighbor_ec, str) or not '.' in neighbor_ec:
        raise ValueError("Invalid EC number format.")
    if not isinstance(true_ec, str) or not '.' in true_ec:
        raise ValueError("Invalid EC number format.")
    
    levels1 = neighbor_ec.split(".")
    levels2 = true_ec.split(".")
    alphaandbetabet = 0
    alpha = len(levels1)
    beta = len(levels2)
    for i in range(min(len(levels1), len(levels2))):
        if levels1[i] == levels2[i]:
            alphaandbetabet += 1
        else:
            break
    return alphaandbetabet, alpha, beta

def hiclass_score_metrics(true_ecs: list[str], neighboring_ecs: list[list[str]]) -> tuple[float, float, float]:
    """Calculates HiCLASS score metrics for a set of true EC numbers and their neighboring EC numbers.

    The HiCLASS (Hierarchical Classification) score metrics are used to evaluate the
    performance of classification algorithms in a hierarchical structure. In the context of
    enzyme classification, these metrics assess how well predicted EC numbers (neighboring_ecs)
    match the true EC numbers (true_ecs).

    Args:
        true_ecs (list[str]): A list of true Enzyme Commission (EC) numbers (e.g., ["1.2.3.4", "5.6.7.8"]).
        neighboring_ecs (list[list[str]]): A list of lists of neighboring EC numbers for each true EC number.
            Each inner list contains candidate EC numbers predicted for the corresponding true EC number.

    Returns:
        tuple[float, float, float]: A tuple containing the following HiCLASS score metrics:
            - hP (float): HiCLASS precision, representing the proportion of correctly classified EC numbers.
            - hR (float): HiCLASS recall, representing the completeness of classifications.
            - hF (float): HiCLASS F1-score, the harmonic mean of precision and recall.

    Raises:
        ValueError: If the lengths of `true_ecs` and `neighboring_ecs` do not match,
            indicating a mismatch between the number of true EC numbers and their corresponding neighboring EC numbers.

    Examples:
        >>> true_ecs = ["1.2.3.4", "5.6.7.8"]
        >>> neighboring_ecs = [["1.2.3.4", "1.2.3.5"], ["5.6.7.7", "5.6.7.9"]]
        >>> hP, hR, hF = hiclass_score_metrics(true_ecs, neighboring_ecs)
        >>> print(f"HiCLASS precision (hP): {hP:.4f}")
        >>> print(f"HiCLASS recall (hR): {hR:.4f}")
        >>> print(f"HiCLASS F1-score (hF): {hF:.4f}")
    """
    if len(true_ecs) != len(neighboring_ecs):
        raise ValueError("Length mismatch between true_ecs and neighboring_ecs lists.")

    intersection = 0
    alpha = 0
    beta = 0
    for true_ec , neighbor_ecs in zip(true_ecs, neighboring_ecs):
        max_score = 0
        for neighbor_ec in neighbor_ecs:
            alphaandbetabet, a, b = overlap_score(neighbor_ec, true_ec)
            max_score = max(max_score, alphaandbetabet)
        intersection += max_score
        alpha += a
        beta += b
    hP = intersection/alpha
    hR = intersection/beta
    hF = 2*hP*hR/(hP+hR)
    return  hP, hR, hF

def level_accuracy(accuracies: dict[int, int], ec: str, retrieved_ec: str) -> dict[int, int]:
    """Calculates accuracy at different levels of the Enzyme Commission (EC) number hierarchy.

    This function updates a dictionary `accuracies` that tracks the number of correctly
    retrieved EC numbers at each level (key) in the hierarchy. It iterates through the
    levels in the `accuracies` dictionary and checks if the first `level` levels (inclusive)
    of the true EC number (`ec`) match the corresponding levels of a retrieved EC number (`retrieved_ec`).

    Args:
        accuracies (dict[int, int]): A dictionary where keys are levels (integers) and
            values are the number of correctly retrieved EC numbers at that level.
        ec (str): The true EC number (e.g., "1.2.3.4").
        retrieved_ec (str): A List of retrieved EC number to be compared against the true EC number.

    Returns:
        dict[int, int]: The updated `accuracies` dictionary with potentially incremented counts
            for correctly retrieved EC numbers at different levels.
    """
#     print(ec, retrieved_ec)
    for level, accuracy in accuracies.items():
        for r_ec in retrieved_ec:
            if ec.split('.')[:level] == r_ec.split('.')[:level]:
                accuracies[level] += 1
                break
    return accuracies


def top_k_retrieval(train_ec: list[str], test_ec: list[str], train_embeddings: torch.Tensor, test_embeddings: torch.Tensor) -> dict[int, tuple[dict[int, float], tuple[float, float, float]]]:
    """
    Performs top-k retrieval of EC numbers using Faiss and calculates accuracy and HiCLASS metrics.

    This function retrieves the top-k nearest neighbors (most similar EC numbers) for
    each EC number in the test set using Faiss for efficient nearest neighbor search.
    It then calculates accuracy at different levels of the EC number hierarchy and
    HiCLASS score metrics to evaluate the retrieval performance.

    Args:
        train_ec (list[str]): A list of true EC numbers in the training set.
        test_ec (list[str]): A list of true EC numbers for which to retrieve neighbors.
        train_embeddings (torch.Tensor): A tensor of embeddings for the training set EC numbers.
            The tensor is expected to have dimensions (num_train_ec, embedding_dim).
        test_embeddings (torch.Tensor): A tensor of embeddings for the test set EC numbers.
            The tensor is expected to have dimensions (num_test_ec, embedding_dim).

    Returns:
        dict[int, tuple[dict[int, float], tuple[float, float, float]]]: A dictionary where keys are
            k values (e.g., 1, 3, 5) and values are tuples containing:
                - accuracies (dict[int, float]): A dictionary where keys are levels (integers)
                    and values are the accuracy (proportion of correctly retrieved EC numbers)
                    at that level.
                - hiclass_metric (tuple[float, float, float]): A tuple containing HiCLASS score metrics
                    (precision, recall, F1-score) for the top-k retrieval results.
    """
    print(f"Data dimensions: train_ec={len(train_ec)}, test_ec={len(test_ec)}, "
          f"train_embeddings={train_embeddings.size()}, test_embeddings={test_embeddings.size()}")
    results = {}
    index = faiss.IndexFlatL2(train_embeddings.size(1))  # Use L2 distance for cosine similarity
    index.add(train_embeddings.cpu().detach().numpy())  # Add embeddings to the index
    K = [1,3,5]
    for k in K:
        distances, retrieval_indices = index.search(test_embeddings.cpu().detach().numpy(), k)
        retrieval_results = []
        for i, retrieved_idxs in enumerate(retrieval_indices):
    #         print(retrieved_idxs)
            retrieved_ec_numbers = [train_ec[idx] for idx in retrieved_idxs]
            retrieval_results.append(retrieved_ec_numbers)
        accuracies = {} # initialize ec level accuracy
        for level in range(4):
            accuracies.setdefault(level+1, 0)
        for i, retrieved_ec_numbers in enumerate(retrieval_results):
            accuracies = level_accuracy(accuracies, test_ec[i], retrieved_ec_numbers)
        for level, total_correct in accuracies.items():
            accuracies[level] /= len(test_ec)
        #calculate hiclass metrics
        hiclass_metric = hiclass_score_metrics(test_ec, retrieval_results)
        results[k] = accuracies, hiclass_metric
    return results


def get_embedding_stats(embeddings: torch.Tensor):
    """
    This function extracts class token embedding, minimum, maximum and average 
    for each sequence from the encoder outputs.
    Args:
        embeddings (torch.Tensor): Encoder outputs of shape (batch_size, sequence_length, embedding_dim)
    Returns:
        tuple: A tuple containing three tensors:
            - class_token_embeddings (torch.Tensor): Class token embeddings of shape (batch_size, embedding_dim)
            - min_embeddings (torch.Tensor): Minimum embedding per sequence of shape (batch_size, embedding_dim)
            - max_embeddings (torch.Tensor): Maximum embedding per sequence of shape (batch_size, embedding_dim)
            - avg_embeddings (torch.Tensor): Average embedding per sequence of shape (batch_size, embedding_dim)
    """
    class_token_embeddings = embeddings[:, 0, :]# Extract class token embeddings (assuming it's at index 0 for each sequence)
    min_embeddings, _ = torch.min(embeddings, dim=1)
    max_embeddings, _ = torch.max(embeddings, dim=1)
    avg_embeddings = torch.mean(embeddings, dim=-2) # torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
    return class_token_embeddings, min_embeddings, max_embeddings, avg_embeddings

def load_df(data_dir: str, file_prefix: str) -> pd.DataFrame:
    """Loads dataframes from CSV files matching a specific prefix within a directory.

    This function searches for CSV files with a given prefix (`file_prefix`)
    within a specified directory (`data_dir`) and combines them into a single
    pandas DataFrame.

    Args:
        data_dir (str): The directory path containing the CSV files.
        file_prefix (str): The prefix of the CSV files to load (e.g., "train", "test").

    Returns:
        pd.DataFrame: A combined DataFrame containing data from all loaded CSV files.

    Raises:
        FileNotFoundError: If no files are found matching the specified prefix and directory.
    """
    # data_dir = "/work/ah2lab/soumya/"
    file_ptrn = f"{file_prefix}*"
    all_files = glob.glob(f"{data_dir}/{file_ptrn}.csv")
    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern '{file_prefix}' in directory '{data_dir}'.")
    print(f"Loading data from files: {all_files}")  # Informative message
    df= pd.DataFrame()   #load train and test separately
    for filename in all_files:
        temp_df = pd.read_csv(filename) #, usecols=['EC','Sequence']
        df = pd.concat([df, temp_df], ignore_index=True)
    return df

# def get_embedding_stats(embeddings):
#     min_embeddings, _ = torch.min(embeddings, dim=1)
#     max_embeddings, _ = torch.max(embeddings, dim=1)
#     avg_embeddings = torch.mean(embeddings, dim=-2) # torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
#     return embeddings[:, 0, :], min_embeddings, max_embeddings, avg_embeddings


def compute_cosine_similarity_in_chunks(embeddings: torch.Tensor, chunk_size=1000) -> torch.Tensor:
    """Computes cosine similarity matrix for a large embedding tensor in chunks for memory efficiency.

    This function calculates the pairwise cosine similarity between all rows (vectors)
    in the input embedding tensor (`embeddings`). Due to memory constraints when dealing
    with large datasets, the computation is performed in chunks to avoid loading the
    entire matrix into memory at once.

    Args:
        embeddings (torch.Tensor): A 2D tensor of embeddings (num_embeddings x embedding_dim).
        chunk_size (int, optional): The size of each chunk to process. Defaults to 1000.

    Returns:
        torch.Tensor: A 2D tensor of cosine similarities between all embedding pairs (num_embeddings x num_embeddings).
    """
    n = embeddings.size(0)
    similarity_matrix = torch.zeros((n, n))
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            end_j = min(j + chunk_size, n)
            chunk1 = embeddings[i:end_i]
            chunk2 = embeddings[j:end_j]
            similarity_chunk = torch.nn.functional.cosine_similarity(chunk1.unsqueeze(1), chunk2.unsqueeze(0), dim=2)
            similarity_matrix[i:end_i, j:end_j] = similarity_chunk.cpu()
    return similarity_matrix

def get_embeddings(tokenizer, model, loader: torch.utils.data.DataLoader, device: torch.device, train_sequences: list[str], batch_size: int) -> torch.Tensor:
    """
    Extracts embeddings for a list of sequences using a pre-trained tokenizer and model.

    This function takes a list of protein or DNA sequences (`train_sequences`), a tokenizer
    (`tokenizer`), a pre-trained model (`model`), a data loader (`loader`), and the target device
    (`device`) as input. It iterates through the data loader in batches, performs tokenization
    and padding according to the tokenizer's configuration for the specific model type, and
    extracts the relevant hidden state from the model's outputs to represent the embeddings.
    The embeddings for all sequences are then concatenated and returned.

    Args:
        tokenizer : The tokenizer to be used for tokenization and padding.
        model : The pre-trained model to extract embeddings from.
        loader (torch.utils.data.DataLoader): The data loader for iterating through batches.
        device (torch.device): The device (CPU or GPU) to use for computations.
        train_sequences (list[str]): A list of protein or DNA sequences for which to extract embeddings.
        batch_size (int): The batch size for processing sequences.

    Returns:
        torch.Tensor: A tensor of concatenated embeddings for all input sequences
            (num_sequences x embedding_dim).
    """
    if 'LOLBERT' in tokenizer.name_or_path or 'FinetunedModel' in tokenizer.name_or_path:
        train_batch = tokenizer(train_sequences, padding='max_length', truncation=True) #lolbert
    elif 'DNABERT' in tokenizer.name_or_path:
        train_batch = tokenizer(train_sequences, max_length=512, return_tensors='pt', truncation=True, padding=True) #dnabert
    else:
        train_batch = tokenizer.batch_encode_plus(train_sequences, max_length=128, return_tensors='pt', truncation=True, padding=True) #nuc
    
    train_encodings = {'input_ids': torch.tensor(train_batch['input_ids']), 'attention_mask': torch.tensor(train_batch['attention_mask'])} #, 'labels': torch.tensor(batch['input_ids'])
    train_dataset = Dataset(train_encodings)
    torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # loop = tqdm(loader, leave=True)
    embeddings = []#torch.Tensor()#.to(device)
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)#     labels = batch['labels'].to(device)
        with torch.no_grad():
            if 'DNABERT' in tokenizer.name_or_path or 'LOLBERT' in tokenizer.name_or_path or 'FinetunedModel' in tokenizer.name_or_path:
                outputs = model(input_ids, attention_mask=attention_mask) # for lol and dna bert
            else:
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True) # for nuc trans
        if 'LOLBERT' in tokenizer.name_or_path:
            batch_embeddings = outputs.last_hidden_state.cpu() # LOLBERT
        elif 'DNABERT' in tokenizer.name_or_path:
            batch_embeddings = outputs.hidden_states.cpu() #dnabert
        else:
            batch_embeddings = outputs['hidden_states'][-1].cpu() #for nuc trans
        embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings

def dist_matrix_chunks(embeddings: np.ndarray, chunk_size=1000) -> np.ndarray:
    """
    Computes pairwise cosine distance matrix for a large embedding array in chunks for memory efficiency.

    This function calculates the cosine distance between all pairs of rows (vectors)
    in the input embedding array (`embeddings`). Due to memory constraints when dealing
    with large datasets, the computation is performed in chunks to avoid loading the
    entire matrix into memory at once.

    Args:
        embeddings (np.ndarray): A 2D NumPy array of embeddings (num_embeddings x embedding_dim).
        chunk_size (int, optional): The size of each chunk to process. Defaults to 1000.

    Returns:
        np.ndarray: A 2D NumPy array of cosine distances between all embedding pairs (num_embeddings x num_embeddings).
    """

    n = embeddings.shape[0]
    distance_matrix = np.zeros((n, n))
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            end_j = min(j + chunk_size, n)
            chunk1 = embeddings[i:end_i]
            chunk2 = embeddings[j:end_j]
            distance_chunk = cosine(embeddings[i].cpu().detach().numpy(), embeddings[j].cpu().detach().numpy())
            distance_matrix[i:end_i, j:end_j] = distance_chunk
    return distance_matrix


def load_tokenizer(model_name: str) -> BioTokenizer:
  """Loads a pre-trained tokenizer for protein or DNA sequences.

    This function loads a tokenizer from the Transformers library based on the provided model name.
    It's likely that the `BioTokenizer` class is a custom class that modifies the tokenization
    behavior for your specific needs. The function retrieves the tokenizer from the `models` subdirectory
    and sets the vocabulary tokens (CLS, SEP, PAD, etc.) based on pre-defined values.

    **Note:** It's important to ensure that the `BioTokenizer` class appropriately handles the
             intended tokenization and maps the provided vocabulary tokens to the model's requirements.

    Args:
        model_name (str): The name of the pre-trained model to load the tokenizer for (e.g., "genebert").

    Returns:
        transformers.AutoTokenizer: The loaded tokenizer for the specified model.
    """
  tokenizer = BioTokenizer.from_pretrained(f"models/{model_name}", max_len=128) #genebert is of no use actually, since Biotokenizer class is overwritting tokenize function
  cls_token = "S"
  pad_token = "P"
  sep_token = "/S"
  unk_token = "N"
  mask_token = "M"
  G_token = "G"
  A_token = "A"
  C_token = "C"
  T_token = "T"
  token_ids = tokenizer.convert_tokens_to_ids([cls_token, pad_token, sep_token, unk_token, mask_token, G_token, A_token, C_token, T_token])
  tokenizer.cls_token_id = token_ids[0]
  tokenizer.pad_token_id = token_ids[1]
  tokenizer.sep_token_id = token_ids[2]
  tokenizer.unk_token_id = token_ids[3]
  tokenizer.mask_token_id = token_ids[4]
  tokenizer.bos_token_id = token_ids[0]
  tokenizer.eos_token_id = token_ids[2]
  return tokenizer

class Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset class for protein or DNA sequence embeddings.

    This class inherits from `torch.utils.data.Dataset` to represent a custom dataset
    specifically designed for protein or DNA sequence embeddings. It stores the encoded
    sequences (`encodings`) internally and provides methods to access them during data
    loader iteration.

    Args:
        encodings (dict[str, torch.Tensor]): A dictionary containing pre-processed and encoded sequences.
            The expected keys are:
                - 'input_ids': A tensor of input token IDs.
                - 'attention_mask': A tensor of attention masks.
                (Optional)
                - 'labels': A tensor of labels (if applicable).
    """
    def __init__(self, encodings):
        self.encodings = encodings
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        This method overrides the default behavior of `__len__` to return the number of
        sequences (samples) present in the `encodings` dictionary.
        """
        return self.encodings['input_ids'].shape[0]
    def __getitem__(self, i):
        """
        Retrieves a single sample from the dataset at the specified index.

        This method overrides the default behavior of `__getitem__` to return a dictionary
        containing the following tensors for the given index `i`:
            - 'input_ids': The input token IDs for the i-th sample.
            - 'attention_mask': The attention mask for the i-th sample.
            (Optional)
            - 'labels': The label for the i-th sample (if applicable).
        """
        return {key: tensor[i] for key, tensor in self.encodings.items()}
    