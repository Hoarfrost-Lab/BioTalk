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


def overlap_score(neighbor_ec, true_ec):
    """
    This function calculates the overlap score between two EC numbers.
    Args:
        ec1 (str): The first EC number (e.g., "1.2.3.4").
        ec2 (str): The second EC number (e.g., "1.2.3.5").
    Returns:
        int: The overlap score, which is the number of matching levels from the beginning.
    """
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

def hiclass_score_metrics(true_ecs, neighboring_ecs):
    """
    This function calculates the hierarchical precision (hP) for a set of predicted EC numbers.
    Args:
        true_ecs (list): A list of true EC numbers (e.g., ["1.2.3.4", "2.3.4.5"]).
        neighboring_ecs (list): A list of predicted EC numbers, where each element is a list of k nearest neighbors (e.g., [["1.2.3.5", "2.1.4.3"], ["3.4.5.6", "2.3.4.1"]]).
    Returns:
        float: The average hierarchical precision across all predicted EC numbers. 
    """
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

def level_accuracy(accuracies, ec , retrieved_ec):
#     print(ec, retrieved_ec)
    for level, accuracy in accuracies.items():
        for r_ec in retrieved_ec:
            if ec.split('.')[:level] == r_ec.split('.')[:level]:
                accuracies[level] += 1
                break
    return accuracies
def top_k_retrieval(train_ec, test_ec, train_embeddings, test_embeddings):
    print(f"train_ec={len(train_ec)}, test_ec={len(test_ec)}, train_embeddings={len(train_embeddings)}, test_embeddings={len(test_embeddings)}")
    """
    Performs top-k NN retrieval using RoBERTa encoder embeddings and evaluates
    EC number consistency in top-k retrievals.
    Args:
        model_name (str): Name of the RoBERTa model (e.g., 'roberta-base').
        test_ec (list): List of known EC numbers of test batch.
        train_seqs (list): List of sequences data for which to retrieve similar sequences.
        test_seqs (list): List of sequences data against which retrivals are called.
        k (int): Number of nearest neighbors to retrieve.
    Returns:
        tuple: (accuracy, retrieval_results)
            - accuracy (float): Proportion of top-k retrievals with correct EC numbers.
            - retrieval_results (list of lists): List of retrieved EC numbers for each text data point.
    """
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
def get_embedding_stats(embeddings):
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

def load_df(data_dir, file_prefix):
    # data_dir = "/work/ah2lab/soumya/"
    file_ptrn = f"{file_prefix}*"
    all_files = glob.glob(f"{data_dir}/{file_ptrn}.csv")
    print(all_files)
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


def compute_cosine_similarity_in_chunks(embeddings, chunk_size=1000):
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

def get_embeddings(tokenizer, model, loader, device, train_sequences, batch_size):
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

def dist_matrix_chunks(embeddings, chunk_size=1000):
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


def load_tokenizer(model_name):
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
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings
    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]
    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}
    