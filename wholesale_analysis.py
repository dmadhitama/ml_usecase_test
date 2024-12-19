import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import sys
import os
import argparse
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

# Model version
MODEL_VERSION = "1.0.0"

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

if not os.path.exists("logs"):
    os.makedirs("logs")
logger.add("logs/wholesale_analysis_{time}.log", rotation="500 MB")

class WholesaleDataset(Dataset):
    """Custom Dataset for wholesale customers data."""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SimpleNN(nn.Module):
    """Simple Neural Network for classification."""
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.3)
        )
        self.layer4 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def analyze_data_distribution(df):
    """Analyze the distribution of features and target."""
    logger.info("Analyzing data distribution")
    
    # Check class distribution
    class_dist = df['Region'].value_counts()
    logger.info("\nClass distribution:")
    for region, count in class_dist.items():
        logger.info(f"Region {region}: {count} samples ({count/len(df)*100:.2f}%)")
    
    # Check feature statistics
    feature_cols = [col for col in df.columns if col not in ['Region']]
    stats = df[feature_cols].describe()
    logger.info("\nFeature statistics:")
    logger.info(f"\n{stats}")
    
    # Check for outliers
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)][col]
        if len(outliers) > 0:
            logger.warning(f"Found {len(outliers)} outliers in {col}")

def load_and_preprocess_data(file_path):
    """Load and preprocess the wholesale customers dataset."""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {len(df)} rows of data")
        
        logger.info("DataFrame columns:")
        for col in df.columns:
            logger.info(f"Column: '{col}' - Type: {df[col].dtype}")
        
        # Analyze data distribution
        analyze_data_distribution(df)
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_data(df):
    """Prepare features and target variables."""
    logger.info("Preparing features and target variables")
    try:
        # Separate features and target
        X = df.drop(['Region'], axis=1)
        y = df['Region'].values - 1  # Convert to numpy array and make 0-based
        
        # Use RobustScaler for better handling of outliers
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check scaled data statistics
        logger.info("\nScaled features statistics:")
        scaled_stats = pd.DataFrame(X_scaled, columns=X.columns).describe()
        logger.info(f"\n{scaled_stats}")
        
        # Split the data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Convert to numpy arrays with specific dtypes
        X_train = np.array(X_train, dtype=np.float32)
        X_test = np.array(X_test, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int64)
        y_test = np.array(y_test, dtype=np.int64)
        
        logger.info(f"Features: {', '.join(X.columns)}")
        logger.info(f"Number of classes: {len(np.unique(y))}")
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Log class distribution in train and test sets
        for split_name, y_data in [("Train", y_train), ("Test", y_test)]:
            unique, counts = np.unique(y_data, return_counts=True)
            dist = dict(zip(unique, counts))
            logger.info(f"{split_name} set class distribution: {dist}")
        
        return X_train, X_test, y_train, y_test, X.columns
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train_model(model, train_loader, criterion, optimizer, device, num_epochs):
    """Train the neural network."""
    logger.info("Starting model training")

    # Create directories if they don't exist
    if not os.path.exists('wholesale_analysis'):
        os.makedirs('wholesale_analysis')
    
    if not os.path.exists('models'):
        os.makedirs('models')
        
    try:
        model.train()
        best_loss = float('inf')
        train_losses = []
        train_accuracies = []
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
            
            # Save model if it has the best loss
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                logger.info(f'New best loss: {best_loss:.4f}. Saving model...')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, 'models/best_model.pth')

        # Plot training progress
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig('wholesale_analysis/training_progress.png')
        plt.close()
        
        logger.success("Training completed")
        logger.info(f"Best loss achieved: {best_loss:.4f}")
        return model
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def evaluate_model(model, test_loader, device, num_classes, output_dir):
    """Evaluate the trained model."""
    logger.info("Evaluating model")
    try:
        model.eval()
        metric = MulticlassAccuracy(num_classes=num_classes).to(device)
        conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                metric.update(predicted, labels)
                conf_matrix.update(predicted, labels)
        
        accuracy = metric.compute()
        cm = conf_matrix.compute()
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm.cpu().numpy(), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.grid()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        return accuracy, cm
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def parse_arguments():
    parser = argparse.ArgumentParser(description='Wholesale Customers Region Classification using PyTorch')
    parser.add_argument('--input', '-i', required=True, help='Path to the input CSV file')
    parser.add_argument('--output', '-o', default='analysis_results', help='Directory to save output files (default: analysis_results)')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--hidden-size', '-hs', type=int, default=64, help='Hidden layer size (default: 64)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    logger.info("Starting wholesale customer region classification using PyTorch")
    
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load and prepare data
        df = load_and_preprocess_data(args.input)
        X_train, X_test, y_train, y_test, feature_names = prepare_data(df)
        
        # Create data loaders
        train_dataset = WholesaleDataset(X_train, y_train)
        test_dataset = WholesaleDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Initialize model
        input_size = len(feature_names)
        num_classes = len(np.unique(y_train))
        model = SimpleNN(input_size, args.hidden_size, num_classes).to(device)
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Train model
        model = train_model(model, train_loader, criterion, optimizer, device, args.epochs)
        
        # Evaluate model
        accuracy, confusion_matrix = evaluate_model(model, test_loader, device, num_classes, args.output)
        
        logger.success("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
