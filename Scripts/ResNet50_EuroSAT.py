# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import os
import pandas as pd
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

model_name = 'ResNet50'

def train_model():
    # Random seed: 452
    set_seed(452)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set paths
    data_dir = "./Processed_EuroSAT_64x64"

    # Data augmentation and transforms
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), 
        transforms.RandomRotation(90),    
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transforms)


    # Calculate class weights to handle imbalance
    class_counts = [
        len([entry for entry in os.scandir(os.path.join(data_dir, class_name)) if entry.is_file()]) 
        for class_name in full_dataset.classes
    ]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float).to(device)

    # Splitting the dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create a mapping from indices to class labels for the train dataset
    train_labels = [full_dataset.targets[idx] for idx in train_dataset.indices]
    sample_weights = [class_weights[label] for label in train_labels]
    
    # Weighted random sampler for imbalanced classes
    train_sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    # Data loaders
    train_loader = DataLoader(
        Subset(full_dataset, train_dataset.indices), 
        batch_size=32, 
        sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_dataset.indices),
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # Load ResNet50 with pretrained weights
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(full_dataset.classes))
    
    model = model.to(device)

    # Use CrossEntropyLoss with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # Cyclic LR Scheduler
    scheduler = CyclicLR(
        optimizer, 
        base_lr=1e-6, 
        max_lr=1e-4, 
        step_size_up=5, 
        mode='triangular2'
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # Early stopping parameters
    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_f1 = float('-inf')
    min_delta = 1e-4

    # Directory to save models
    model_save_dir = f"{model_name}_EuroSAT"
    os.makedirs(model_save_dir, exist_ok=True)

    # List to store metrics for CSV
    metrics = []

    num_epochs = 75

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_targets = []
        train_outputs = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            train_outputs.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader.dataset)
        train_outputs = np.array(train_outputs)
        train_targets = np.array(train_targets)
        
        # Training metrics
        train_f1 = f1_score(train_targets, train_outputs, average='macro')
        train_precision = precision_score(train_targets, train_outputs, average='macro')
        train_recall = recall_score(train_targets, train_outputs, average='macro')
        train_accuracy = accuracy_score(train_targets, train_outputs)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_targets = []
        val_outputs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                val_outputs.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_outputs = np.array(val_outputs)
        val_targets = np.array(val_targets)

        # Validation metrics
        val_f1 = f1_score(val_targets, val_outputs, average='macro')
        val_precision = precision_score(val_targets, val_outputs, average='macro')
        val_recall = recall_score(val_targets, val_outputs, average='macro')
        val_accuracy = accuracy_score(val_targets, val_outputs)

        # Print logs
        print(f"Epoch {epoch}/{num_epochs}")
        print('-' * 10)
        print(
            f"train Loss: {train_loss:.4f} "
            f"Acc: {train_accuracy:.4f} "
            f"Recall: {train_recall:.4f} "
            f"Precision: {train_precision:.4f} "
            f"F1: {train_f1:.4f}"
        )
        print(
            f"val   Loss: {val_loss:.4f} "
            f"Acc: {val_accuracy:.4f} "
            f"Recall: {val_recall:.4f} "
            f"Precision: {val_precision:.4f} "
            f"F1: {val_f1:.4f}"
        )

        # Save metrics to list for CSV
        metrics.append({
            'Phase': 'train',
            'Loss': train_loss,
            'Accuracy': train_accuracy,
            'Recall': train_recall,
            'Precision': train_precision,
            'F1': train_f1
        })
        metrics.append({
            'Phase': 'val',
            'Loss': val_loss,
            'Accuracy': val_accuracy,
            'Recall': val_recall,
            'Precision': val_precision,
            'F1': val_f1
        })

        # Early stopping check
        if (val_f1 - best_val_f1) > min_delta:
            best_val_f1 = val_f1
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        # Step the scheduler
        scheduler.step()

        # Save model at each epoch
        torch.save(model.state_dict(), os.path.join(model_save_dir, f'Epoch_{epoch}_{model_name}.pth'))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(model_save_dir, f'{model_name}_Metrics_EuroSAT.csv'), index=False)

    print("Training complete.")

if __name__ == "__main__":
    train_model()