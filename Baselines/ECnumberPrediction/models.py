import torch
import torch.nn as nn
import torch.optim as optim

class TwoLayerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TwoLayerClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

def setup_network(input_size, hidden_size, num_classes):
    """
    Setup the neural network, loss criterion, and optimizer.

    Args:
        input_size (int): The size of the input features (size of the embeddings).
        hidden_size (int): The size of the hidden layer.
        num_classes (int): The number of classes for classification.

    Returns:
        dict: A dictionary containing the model, criterion, and optimizer.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TwoLayerClassifier(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # Suitable for multiclass classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Using Adam optimizer

    return {'model': model, 'criterion': criterion, 'optimizer': optimizer, 'device': device}
