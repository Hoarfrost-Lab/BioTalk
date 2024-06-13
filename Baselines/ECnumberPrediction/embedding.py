import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, RobertaModel
from transformers.models.bert.configuration_bert import BertConfig
from utils import utils

class EmbeddingGenerator:
    def __init__(self):
        """
        Initializes the embedding generators for DNABERT, nucleotide transformer models and LOLBERT4.
        """
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # DNABERT model and tokenizer
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.dnabert_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
        self.dnabert_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
        self.dnabert_model.eval()  # Set the DNABERT model to evaluation mode
        
        # Nucleotide Transformer model and tokenizer
        self.nutransformer_tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        self.nutransformer_model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")
        self.nutransformer_model.eval()  # Set the nucleotide transformer model to evaluation mode

        # LOLBERT4 model and tokenizer
        self.lolbert4_tokenizer = utils.load_tokenizer("LOLBERTv4")
        self.lolbert4_model = RobertaModel.from_pretrained("models/LOLBERTv4/")
        self.lolbert4_model.eval()

    def generate_dnabert_embeddings(self, dna_sequence):
        """
        Generate embeddings for a given DNA sequence using the DNABERT model.
        
        Args:
            dna_sequence (str): The DNA sequence to encode.
        
        Returns:
            torch.Tensor: The mean-pooled embeddings produced by the DNABERT model.
        """
        inputs = self.dnabert_tokenizer(dna_sequence, return_tensors='pt')["input_ids"]
        hidden_states = self.dnabert_model(inputs)[0]  # Get the hidden states
        embedding_mean = torch.mean(hidden_states[0], dim=0)  # Mean pooling
        return embedding_mean

    def generate_nutransformer_embeddings(self, sequences):
        """
        Generate mean-pooled embeddings for a list of DNA sequences using the nucleotide transformer model.
        
        Args:
            sequences (list of str): The list of DNA sequences to encode.
        
        Returns:
            torch.Tensor: The mean-pooled embeddings produced by the nucleotide transformer.
        """
        # check if input is a single string, wrap it in a list if true
        if isinstance(sequences, str):
            sequences = [sequences]

        max_length = self.nutransformer_tokenizer.model_max_length
        tokens_ids = self.nutransformer_tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length=max_length)["input_ids"]
        attention_mask = tokens_ids != self.nutransformer_tokenizer.pad_token_id
        torch_outs = self.nutransformer_model(tokens_ids, attention_mask=attention_mask, encoder_attention_mask=attention_mask, output_hidden_states=True)
        embeddings = torch_outs['hidden_states'][-1].detach()  # Use the last layer
        attention_mask = torch.unsqueeze(attention_mask, dim=-1)  # Add embed dimension axis
        mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2) / torch.sum(attention_mask, axis=1)  # Compute mean embeddings per sequence
        # squeeze the tensor to remove the extra dimension
        mean_sequence_embeddings = torch.squeeze(mean_sequence_embeddings)
        return mean_sequence_embeddings
    
    def generate_lolbert4_embeddings(self, sequences):
        """
        Generate embeddings for a given DNA sequence using the LOLBERTv4 model.
        
        Args:
            dna_sequence (str): The DNA sequence to encode.
        
        Returns:
            torch.Tensor: The mean-pooled embeddings produced by the LOLBERTv4 model.
        """
        inputs = self.lolbert4_tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)
        hidden_states = self.lolbert4_model(**inputs).last_hidden_state
        embedding_mean = torch.mean(hidden_states, dim=1).squeeze(0)
        return embedding_mean
