# Import required libraries
import numpy as np
import os
import torch
import torch.nn as nn
from torch import nn
import seaborn
import pandas as pd
import torch.optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from torchvision import datasets, transforms, models

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set up device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Final neural network model design 
class Ensemble_Neural_Network(nn.Module):
    def __init__(self, input_size=30, hidden_size=64, num_classes=10):
        super(Ensemble_Neural_Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size//2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# Function to load model based on name
def load_model(model_name):
    print(f"Loading model: {model_name}")
    
    if model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
    elif model_name == "RegNet-Y-8GF":
        model = models.regnet_y_8gf(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(full_dataset.classes))
    elif model_name == "EfficientNetV2S":
        model = models.efficientnet_v2_s(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(full_dataset.classes))
    else:
        raise ValueError("FATAL ERROR. MODEL WEIGHTS NOT PRESENT.")
    
    # Load the model weights
    model.load_state_dict(torch.load(model_paths[model_name], map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Function to get predictions from a model
def get_predictions(model, dataloader):
    all_preds = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            all_preds.extend(probabilities.cpu().numpy())
    return np.array(all_preds)

if __name__ == '__main__':
    # Load full dataset
    data_dir = "./Processed_EuroSAT_64x64"
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Class 0: AnnualCrop
    # Class 1: Forest
    # Class 2: HerbaceousVegetation
    # Class 3: Highway
    # Class 4: Industrial
    # Class 5: Pasture
    # Class 6: PermanentCrop
    # Class 7: Residential
    # Class 8: River
    # Class 9: SeaLake

    full_dataset = datasets.ImageFolder(data_dir, transform=val_transforms)
    
    # Model paths
    model_paths = {
        "ResNet50": "./ResNet50_EuroSAT/Epoch_29_ResNet50.pth",
        "RegNet-Y-8GF": "./RegNet-Y-8GF_EuroSAT/Epoch_18_RegNet-Y-8GF.pth",
        "EfficientNetV2S": "./EfficientNetV2-S_EuroSAT/Epoch_27_EfficientNetV2-S.pth"
    }

    all_indices = np.arange(len(full_dataset))
    y = np.array([label for _, label in full_dataset.imgs])
    
    # First split: 80% train+val, 20% final test
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    # Load CNNs and get predictions for test set
    resnet50 = load_model("ResNet50")
    regnet_y_8gf = load_model("RegNet-Y-8GF")
    efficientnetv2s = load_model("EfficientNetV2S")

    # Create test dataset
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=10,
        pin_memory=True
    )

    # Get test predictions (for feature matrix)
    test_resnet50_preds = get_predictions(resnet50, test_loader)
    test_regnet_y_8gf_preds = get_predictions(regnet_y_8gf, test_loader)
    test_efficientnetv2s_preds = get_predictions(efficientnetv2s, test_loader)

    # Create train+val loader
    train_val_dataset = torch.utils.data.Subset(full_dataset, train_val_indices)
    train_val_loader = torch.utils.data.DataLoader(
        train_val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=10,
        pin_memory=True
    )

    # Get train+val predictions (for feature matrix)
    train_val_resnet50_preds = get_predictions(resnet50, train_val_loader)
    train_val_regnet_y_8gf_preds = get_predictions(regnet_y_8gf, train_val_loader)
    train_val_efficientnetv2s_preds = get_predictions(efficientnetv2s, train_val_loader)

    # Combine CNN predictions (feature matrix)
    X_test = np.column_stack((
        test_resnet50_preds.reshape(len(test_indices), -1),
        test_regnet_y_8gf_preds.reshape(len(test_indices), -1),
        test_efficientnetv2s_preds.reshape(len(test_indices), -1)
    ))
    y_test = y[test_indices]

    # Combine CNN predictions (feature matrix)
    X_train_val = np.column_stack((
        train_val_resnet50_preds.reshape(len(train_val_indices), -1),
        train_val_regnet_y_8gf_preds.reshape(len(train_val_indices), -1),
        train_val_efficientnetv2s_preds.reshape(len(train_val_indices), -1)
    ))
    y_train_val = y[train_val_indices]

    # Define directories for final neural network
    models_dir = "./Ensemble_Neural_Network_EuroSAT"
    os.makedirs(models_dir, exist_ok=True)
    
    # Cross-validation setup with 5 folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lists to store fold models and metrics
    fold_models = []
    cv_f1_scores = []
    cv_precision_scores = []
    cv_recall_scores = []
    cv_accuracy_scores = []
    cv_loss_scores = []
    cv_auc_scores = [] 

    # Training loop with cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
        print(f"\nFold {fold + 1}")
        print('-' * 150)
        
        # Prepare data for this fold
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        # Create datasets and loaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False
        )
        
        # Final neural network training set up 
        model = Ensemble_Neural_Network().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        best_val_f1 = 0
        patience = 5
        patience_counter = 0
        
        # Training
        for epoch in range(50):
            model.train()
            train_loss = 0.0
            train_predictions = []
            train_labels = []
            train_probs = [] 
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                probs = torch.softmax(outputs, dim=1)
                train_loss += loss.item() * batch_features.size(0)
                train_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_labels.extend(batch_labels.cpu().numpy())
                train_probs.extend(probs.detach().cpu().numpy())
            
            train_loss /= len(train_loader.dataset)
            train_probs = np.array(train_probs)
            train_labels = np.array(train_labels)
            
            # Calculate training metrics
            train_accuracy = accuracy_score(train_labels, train_predictions)
            train_f1 = f1_score(train_labels, train_predictions, average='macro')
            train_precision = precision_score(train_labels, train_predictions, average='macro')
            train_recall = recall_score(train_labels, train_predictions, average='macro')
            
            # Calculate multi-class AUC (using one-vs-rest approach)
            train_auc = roc_auc_score(
                train_labels, 
                train_probs,
                multi_class='ovr',
                average='macro'
            )
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_predictions = []
            val_labels = []
            val_probs = []  
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    probs = torch.softmax(outputs, dim=1)
                    val_loss += loss.item() * batch_features.size(0)
                    val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())
            
            val_loss /= len(val_loader.dataset)
            val_probs = np.array(val_probs)
            val_labels = np.array(val_labels)
            
            # Calculate validation metrics
            val_accuracy = accuracy_score(val_labels, val_predictions)
            val_f1 = f1_score(val_labels, val_predictions, average='macro')
            val_precision = precision_score(val_labels, val_predictions, average='macro')
            val_recall = recall_score(val_labels, val_predictions, average='macro')
            
            # Calculate multi-class AUC (using one-vs-rest approach)
            val_auc = roc_auc_score(
                val_labels, 
                val_probs,
                multi_class='ovr',
                average='macro'
            )
            
            # Print epoch results
            print(f"\nEpoch {epoch}")
            print('-' * 10)
            print(
                f"train Loss: {train_loss:.4f} "
                f"Acc: {train_accuracy:.4f} "
                f"Recall: {train_recall:.4f} "
                f"Precision: {train_precision:.4f} "
                f"F1: {train_f1:.4f} "
                f"AUC: {train_auc:.4f}"
            )
            print(
                f"val   Loss: {val_loss:.4f} "
                f"Acc: {val_accuracy:.4f} "
                f"Recall: {val_recall:.4f} "
                f"Precision: {val_precision:.4f} "
                f"F1: {val_f1:.4f} "
                f"AUC: {val_auc:.4f}"
            )
            
            # Early stopping check
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model for this fold
                torch.save(model.state_dict(), 
                         os.path.join(models_dir, f'Fold_{fold + 1}_Ensemble_Neural_Network.pth'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model for this fold
        model.load_state_dict(torch.load(os.path.join(models_dir, f'Fold_{fold + 1}_Ensemble_Neural_Network.pth')))
        fold_models.append(model)
        
        # Store best validation metrics for this fold
        cv_f1_scores.append(best_val_f1)
        cv_precision_scores.append(val_precision)
        cv_recall_scores.append(val_recall)
        cv_accuracy_scores.append(val_accuracy)
        cv_loss_scores.append(val_loss)
        cv_auc_scores.append(val_auc)

    # Evaluate on test set using all fold models
    test_predictions = []
    test_probs_all = []
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )
    
    # Get predictions from each folds model
    fold_test_metrics = []
    for fold_idx, model in enumerate(fold_models):
        model.eval()
        fold_predictions = []
        fold_probs = []
        
        with torch.no_grad():
            for batch_features, _ in test_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                probs = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                fold_predictions.extend(predictions.cpu().numpy())
                fold_probs.extend(probs.cpu().numpy())
        
        fold_predictions = np.array(fold_predictions)
        fold_probs = np.array(fold_probs)
        test_predictions.append(fold_predictions)
        test_probs_all.append(fold_probs)
        
        # Calculate metrics for this folds predictions
        fold_metrics = {
            'accuracy': accuracy_score(y_test, fold_predictions),
            'f1': f1_score(y_test, fold_predictions, average='macro'),
            'precision': precision_score(y_test, fold_predictions, average='macro'),
            'recall': recall_score(y_test, fold_predictions, average='macro'),
            'auc': roc_auc_score(
                y_test,
                fold_probs,
                multi_class='ovr',
                average='macro'
            )
        }
        fold_test_metrics.append(fold_metrics)

    # Calculate average of test metrics across folds
    test_metrics = {metric: [] for metric in ['accuracy', 'f1', 'precision', 'recall', 'auc']}
    
    for fold_metric in fold_test_metrics:
        for metric in test_metrics:
            test_metrics[metric].append(fold_metric[metric])
    
    # Find best fold based on val F1
    best_fold_idx = np.argmax(cv_f1_scores)
    best_fold_model = fold_models[best_fold_idx]
    print(f"\nBest performing model was from Fold {best_fold_idx + 1} with validation F1: {cv_f1_scores[best_fold_idx]:.4f}")
    
    # Get predictions using existing test loader
    best_fold_predictions = []
    best_fold_labels = []
    best_fold_probabilities = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = best_fold_model(batch_features)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            best_fold_predictions.extend(predictions.cpu().numpy())
            best_fold_probabilities.extend(probabilities.cpu().numpy())
            best_fold_labels.extend(batch_labels.cpu().numpy())

    best_fold_probabilities = np.array(best_fold_probabilities)
    best_fold_labels = np.array(best_fold_labels)

    # Binarize labels
    n_classes = len(full_dataset.classes)
    best_fold_labels_bin = label_binarize(best_fold_labels, classes=range(n_classes))

    # Compute micro-average ROC curve and ROC area
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr["micro"], tpr["micro"], _ = roc_curve(best_fold_labels_bin.ravel(), best_fold_probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([roc_curve(best_fold_labels_bin[:, i], 
                        best_fold_probabilities[:, i])[0] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr_temp, tpr_temp, _ = roc_curve(best_fold_labels_bin[:, i], best_fold_probabilities[:, i])
        mean_tpr += np.interp(all_fpr, fpr_temp, tpr_temp)
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Set style
    plt.style.use('bmh')

    # Plot Micro-Average ROC curve (DIAGRAM 1)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
            label='Micro-Average ROC curve',
            color='#4A90E2', linestyle='-', linewidth=2)

    # Add random classifier line (for 10 classes)
    x = np.linspace(0, 1, 100)
    y = x/10  # For 10 classes
    plt.plot(x, y, 'k--', lw=2, label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Micro-Averaged ROC Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Ensemble_Neural_Network_EuroSAT/Micro_ROC_EuroSAT.png", 
                format='png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Macro-Average ROC curve (DIAGRAM 2)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["macro"], tpr["macro"],
            label='Macro-Average ROC curve',
            color='#E14758', linestyle='-', linewidth=2)

    plt.plot(x, y, 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Macro-Averaged ROC Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Ensemble_Neural_Network_EuroSAT/Macro_ROC_EuroSAT.png", 
                format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    # Create and plot confusion matrix (DIAGRAM 3)
    cm = confusion_matrix(best_fold_labels, best_fold_predictions)
    plt.figure(figsize=(10, 8))

    # Turn off grid
    plt.grid(False)

    seaborn.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.viridis,
                xticklabels=full_dataset.classes,
                yticklabels=full_dataset.classes)
    plt.title('Confusion Matrix on Test Set (Best Fold)', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("./Ensemble_Neural_Network_EuroSAT/Confusion_Matrix_EuroSAT.png")
    plt.close()
    


    # Using best fold model and predictions (DIAGRAM 4)
    class_report = classification_report(best_fold_labels, best_fold_predictions, 
                                      target_names=full_dataset.classes,
                                      output_dict=True)

    # Convert to DataFrame
    df = pd.DataFrame(class_report).transpose()
    df = df.round(4)  # Round to 4 decimal places


    # Rename columns to capitalize
    metrics_df = df.iloc[:-3][['precision', 'recall', 'f1-score']]
    metrics_df.columns = ['Precision', 'Recall', 'F1 Score']

    plt.figure(figsize=(12, 8))

    # Turn off grid
    plt.grid(False)

    # Plot as heatmap 
    seaborn.heatmap(metrics_df, 
                annot=True, fmt='.4f', cmap=plt.cm.cividis,
                yticklabels=full_dataset.classes)
    plt.title('Per-Class Performance Metrics')
    plt.tight_layout()
    plt.savefig("./Ensemble_Neural_Network_EuroSAT/Per_Class_Metrics_EuroSAT.png")
    plt.close()



    # Print cross-validation results
    print("\nCross-Validation Results (validation only):")
    print("-" * 100)
    print(f"Average Loss: {np.mean(cv_loss_scores):.4f} ± {np.std(cv_loss_scores):.4f}")
    print(f"Average Accuracy: {np.mean(cv_accuracy_scores):.4f} ± {np.std(cv_accuracy_scores):.4f}")
    print(f"Average F1 Score: {np.mean(cv_f1_scores):.4f} ± {np.std(cv_f1_scores):.4f}")
    print(f"Average Precision: {np.mean(cv_precision_scores):.4f} ± {np.std(cv_precision_scores):.4f}")
    print(f"Average Recall: {np.mean(cv_recall_scores):.4f} ± {np.std(cv_recall_scores):.4f}")
    print(f"Average AUC Score: {np.mean(cv_auc_scores):.4f} ± {np.std(cv_auc_scores):.4f}")

    # Print final test set results
    print("\nFinal Test Set Results (averaged across folds):")
    print("-" * 100)
    print(
        f"Accuracy: {np.mean(test_metrics['accuracy']):.4f} ± {np.std(test_metrics['accuracy']):.4f}\n"
        f"F1 Score: {np.mean(test_metrics['f1']):.4f} ± {np.std(test_metrics['f1']):.4f}\n"
        f"Precision: {np.mean(test_metrics['precision']):.4f} ± {np.std(test_metrics['precision']):.4f}\n"
        f"Recall: {np.mean(test_metrics['recall']):.4f} ± {np.std(test_metrics['recall']):.4f}\n"
        f"AUC Score: {np.mean(test_metrics['auc']):.4f} ± {np.std(test_metrics['auc']):.4f}"
    )

