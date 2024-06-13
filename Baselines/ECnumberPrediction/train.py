import argparse
import torch
import pandas as pd
from models import setup_network
from dataset import create_data_loader

def get_num_classes(csv_file):
    df = pd.read_csv(csv_file)
    return df['EC'].nunique()  # Assuming 'EC' is the column for labels

def train_model(model, criterion, optimizer, train_loader, device, num_epochs=25, metrics_save_path='training_metrics.json'):
    # Initialize containers to store metrics
    epoch_losses = []
    batch_losses = []
    epoch_accuracies = []
    batch_accuracies = []

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0
        for i, (embeddings, labels) in enumerate(train_loader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # store loss and accuracy
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            batch_accuracies.append(accuracy)

            num_batches += 1

            if i % 100 == 99:
                print(f'Epoch [{epoch+1}, Batch {i+1}], Loss: {loss.item()}')
        
        # store epoch loss and accuracy
        avg_epoch_loss = epoch_loss / num_batches
        epoch_losses.append(avg_epoch_loss)
        epoch_accuracy = 100 * correct / total
        epoch_accuracies.append(epoch_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    print('Finished Training')

    # save metrics to a file
    metrics = {
        'epoch_losses': epoch_losses,
        'batch_losses': batch_losses,
        'epoch_accuracies': epoch_accuracies,
        'batch_accuracies': batch_accuracies
    }
    save_metrics(metrics, metrics_save_path)

    return model

def save_metrics(metrics, path):
    """
    Save the training metrics to a JSON file.

    Args:
        metrics (dict): A dictionary containing the training metrics.
        path (str): The path to the file where metrics should be saved.
    """
    import json
    with open(path, 'w') as f:
        json.dump(metrics, f)
    print(f'Training metrics saved to {path}')

def save_model(model, path):
    """
    Save the model state dictionary.

    Args:
        model (torch.nn.Module): The trained model.
        path (str): The path to save the model file.
    """
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model with embeddings')
    parser.add_argument('--train_csv_path', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--embedding_type', type=str, choices=['dnabert', 'nutransformer','lolbert4'], required=True, help='Type of embedding to use')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--model_save_path', type=str, default='trained_model.pth', help='Path to save the trained model')
    parser.add_argument('--metrics_save_path', type=str, default='training_metrics.json', help='Path to save training metrics')
    parser.add_argument('--save_embeddings', action='store_true', help='Flag to save embeddings')

    args = parser.parse_args()

    input_size = 768 if args.embedding_type in ('dnabert' ,'lolbert4') else 2560
    num_classes = get_num_classes(args.train_csv_path)  # Get dynamic number of classes
    network = setup_network(input_size=input_size, hidden_size=args.hidden_size, num_classes=num_classes)
    model = network['model']
    criterion = network['criterion']
    optimizer = network['optimizer']
    device = network['device']

    train_loader = create_data_loader(args.train_csv_path, args.embedding_type, batch_size=args.batch_size, shuffle=True, save_embeddings=args.save_embeddings, mode='train')

    trained_model = train_model(model, criterion, optimizer, train_loader, device, num_epochs=args.num_epochs, metrics_save_path=args.metrics_save_path)
    
    # Save the model
    save_model(trained_model, args.model_save_path)

# Example usage:
# python train.py --train_csv_path 'train_data.csv' --embedding_type 'dnabert' --num_epochs 10 --batch_size 64 --learning_rate 0.001 --hidden_size 256 --model_save_path 'trained_model.pth' --save_embeddings