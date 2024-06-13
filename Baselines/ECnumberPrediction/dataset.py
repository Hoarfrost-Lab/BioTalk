
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from embedding import EmbeddingGenerator
from utils import encode_labels
import os

class CustomDataset(Dataset):
    def __init__(self, csv_file, embedding_type='dnabert', save_embeddings=False, mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            embedding_type (string): Type of embeddings to generate ('dnabert' or 'nutransformer').
            save_embeddings (bool): Flag to determine whether to save embeddings.
            mode (string): 'train' or 'test' to determine the embedding save directory.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.embedding_generator = EmbeddingGenerator()
        self.embedding_type = embedding_type
        self.label_to_int = encode_labels(self.data_frame['EC'])
        self.save_embeddings = save_embeddings
        self.embeddings_dir = os.path.join(f"{mode}_embeddings",embedding_type)
        
        if self.save_embeddings and not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        sequence = self.data_frame.iloc[idx]['Sequence']
        label_str = self.data_frame.iloc[idx]['EC']
        label = self.label_to_int[label_str]  # Convert label to int

        if self.embedding_type == 'dnabert':
            embedding = self.embedding_generator.generate_dnabert_embeddings(sequence)
        elif self.embedding_type == 'nutransformer':
            embedding = self.embedding_generator.generate_nutransformer_embeddings([sequence])  # Pass as a list
        elif self.embedding_type == 'lolbert4':
            embedding = self.embedding_generator.generate_lolbert4_embeddings(sequence)
        else:
            raise ValueError(f"Invalid embedding type: {self.embedding_type}")
        
        # move the embedding to the GPU if available
        embedding = embedding.to(self.embedding_generator.device)

        if self.save_embeddings:
            embedding_file_path = os.path.join(self.embeddings_dir, f"{idx}.pt")
            torch.save(embedding, embedding_file_path)

        return embedding, torch.tensor(label, dtype=torch.long).to(self.embedding_generator.device)

def create_data_loader(csv_file, embedding_type, batch_size=32, shuffle=True, save_embeddings=False, mode='train'):
    dataset = CustomDataset(csv_file, embedding_type, save_embeddings=save_embeddings, mode=mode)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage
if __name__ == "__main__":
    train_loader = create_data_loader('test_data.csv', 'dnabert', batch_size=64, shuffle=True, save_embeddings=True)
    test_loader = create_data_loader('test_data.csv', 'nutransformer', batch_size=64, shuffle=False, save_embeddings=False)
    
    # To test loading a batch
    for embeddings, labels in train_loader:
        print(embeddings, labels)
        break  # Only print the first batch to check
