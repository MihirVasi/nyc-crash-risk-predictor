import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
MODEL_FILE = "models/crash_predictor.pth"

class CrashDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(np.array(X))
        self.y = torch.LongTensor(np.array(y))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CrashPredictor(nn.Module):
    def __init__(self, input_size):
        super(CrashPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    
    def forward(self, x):
        return self.network(x)

def train_model():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    features = [
        'grid_lat', 'grid_lon', 'hour', 'day_of_week',
        'avg_month', 'is_weekend', 'is_rush_hour', 'is_night',
        'cell_avg', 'cell_max', 'cell_std',
        'hour_avg', 'dow_avg', 'hour_x_dow', 'cell_x_hour'
    ]
    target = 'risk_level'
    
    X_train = train_df[features].values
    y_train = train_df[target].values
    X_test = test_df[features].values
    y_test = test_df[target].values
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(features)}")
    
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights = class_weights / class_weights.sum()
    
    train_dataset = CrashDataset(X_train, y_train)
    test_dataset = CrashDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    model = CrashPredictor(input_size=len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    print("\nTraining model...")
    best_test_acc = 0
    
    for epoch in range(150):
        model.train()
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        scheduler.step()
        
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == y_batch).sum().item()
                test_total += y_batch.size(0)
                all_preds.extend(predicted.numpy())
                all_labels.extend(y_batch.numpy())
        
        train_acc = train_correct / train_total
        test_acc = test_correct / test_total
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'model_state': model.state_dict(),
                'scaler': scaler,
                'features': features
            }, MODEL_FILE)
    
    print(f"\nBest test accuracy: {best_test_acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Low', 'Medium', 'High']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    train_model()