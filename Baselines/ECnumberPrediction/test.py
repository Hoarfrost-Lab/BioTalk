import argparse
import torch
from models import setup_network
from dataset import create_data_loader
from train import get_num_classes
from sklearn.metrics import classification_report
import pandas as pd

def test_model(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    batch_losses = []
    with torch.no_grad():
        for i, (embeddings, labels) in enumerate(test_loader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            # Store batch loss
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            total_loss += batch_loss
            
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    # Calculating metrics
    accuracy = torch.tensor(all_labels).eq(torch.tensor(all_predictions)).sum().item() / len(all_labels)
    report = classification_report(all_labels, all_predictions, output_dict=True)
    
    print(f'Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return all_predictions, {
        'average_loss': avg_loss,
        'accuracy': accuracy * 100,
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score'],
        'classification_report': report
    
    }

def save_predictions_to_csv(predictions, csv_file_path, output_csv_file_path):
    # Load the existing test data
    df = pd.read_csv(csv_file_path)
    
    # Add a new column for predictions
    df['Predicted_Label'] = predictions
    
    # Save the updated DataFrame to a new CSV
    df.to_csv(output_csv_file_path, index=False)    

def save_metrics(metrics, file_path):
    """
    Save metrics to a JSON file.

    Args:
        metrics (dict): A dictionary of metrics to save.
        file_path (str): The path to the file where metrics should be saved.
    """
    import json
    with open(file_path, 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test a model with embeddings')
    parser.add_argument('--test_csv_path', type=str, required=True, help='Path to the test CSV file')
    parser.add_argument('--embedding_type', type=str, choices=['dnabert', 'nutransformer','lolbert4'], required=True, help='Type of embedding to use')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--num_classes', type=int, default=1196, help='Number of classes for testing')
    parser.add_argument('--save_embeddings', action='store_true', help='Flag to save embeddings')
    parser.add_argument('--metrics_save_path', type=str, default='test_metrics.json', help='Path to save test metrics')
    parser.add_argument('--predictions_save_path', type=str, default='updated_test.csv', help='Path to save updated test CSV')
    
    args = parser.parse_args()

    input_size = 768 if args.embedding_type in ('dnabert','lolbert4') else 2560
    num_classes = get_num_classes(args.test_csv_path)  # Ensure consistent num_classes as training

    network = setup_network(input_size=input_size, hidden_size=256, num_classes=args.num_classes)
    model = network['model']
    criterion = network['criterion']
    device = network['device']

    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()

    test_loader = create_data_loader(args.test_csv_path, args.embedding_type, batch_size=64, shuffle=False, save_embeddings=args.save_embeddings, mode='test')
    predictions, test_metrics = test_model(model, criterion, test_loader, device)

    # Save test metrics
    save_metrics(test_metrics, args.metrics_save_path)
    save_predictions_to_csv(predictions, args.test_csv_path, args.predictions_save_path)  # Update CSV with predictions

# Example usage
# python test.py --test_csv_path 'test_data.csv' --embedding_type 'dnabert' --model_path 'trained_model.pth' --num_classes 2228 --save_embeddings
