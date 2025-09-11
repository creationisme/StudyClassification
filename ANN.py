import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

class ClassificationNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
        self.acv = nn.ReLU()
        # self.acv_output = nn.Softmax() : crossentropyloss handles softmax
        self.model_info = False


    def forward(self, x):
        x = self.acv(self.layer1(x))
        x = self.dropout(x)
        x = self.acv(self.layer2(x))
        x = self.dropout(x)
        x = self.acv(self.layer3(x))
        x = self.dropout(x)
        x = self.layer4(x)
        return x
    
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.save_path = save_path
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()
        if self.save_path:
            torch.save(self.best_weights, self.save_path)
            logger.info(f"Saved best model to {self.save_path}")

def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total

def train(input_dim, hidden_dim, output_dim, epochs = 20, patience=5, save_path=None):
    model = ClassificationNN(input_dim, hidden_dim, output_dim) #should it be batch_size instead?
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #can use a learning rate scheduler
    
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    
    early_stopping = EarlyStopping(patience=patience, save_path=save_path)

    logger.info(f"Starting training with {input_dim} features, {output_dim} classes")

    for epoch in range(epochs):
        model.train() #training mode
        curr_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        batch = 0
        for X_batch, y_batch in pbar:

            optimizer.zero_grad()
            outputs = model(X_batch) 
            loss = criterion(outputs, y_batch) #type: torch.Tensor #adding to recognise backward()

            loss.backward() 
            optimizer.step()

            curr_loss += loss.item()
            batch += 1

            pbar.set_postfix({f"Training loss for batch {batch}": f"{loss.item():.4f}" })
        
        train_loss = curr_loss/len(train_dataloader)
        train_acc = calculate_accuracy(model, train_dataloader)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in valid_dataloader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                valid_loss += loss.item()
        
        valid_loss = valid_loss / len(valid_dataloader)
        valid_acc = calculate_accuracy(model, valid_dataloader)
        
        # store metrics
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        logger.info(f'Epoch {epoch+1}/{epochs}:')
        logger.info(f'{'\t'}Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        logger.info(f'{'\t'}Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}{'\n'}')

        if early_stopping(valid_loss, model):
            logger.critical(f'Early stopping triggered after {epoch+1} epochs')
            break

    test_acc = calculate_accuracy(model, test_dataloader)
    logger.info(f'Final Test Accuracy: {test_acc:.4f}')

    if save_path:
        torch.save(model, save_path)
    
    return model, {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accuracies': train_accuracies,
        'valid_accuracies': valid_accuracies
    }

if __name__ == "__main__":
    df = pd.read_csv("NNDataset.csv")
    X = df[df.columns[:-1]]
    y = df["quality"]

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=34)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=34)

    X_train, X_valid, X_test = X_train.to_numpy(), X_valid.to_numpy(), X_test.to_numpy()
    y_train, y_valid, y_test = y_train.to_numpy(), y_valid.to_numpy(), y_test.to_numpy()
    
    train_data = Data(X_train, y_train)
    valid_data = Data(X_valid, y_valid)
    test_data = Data(X_test, y_test)

    batch_size = 32

    train_dataloader = DataLoader(train_data, batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size, shuffle=False)

    logger.info("Setup over for data")
    input_dim = X_train.shape[1] #no. of features
    hidden_dim = 64
    output_dim = len(np.unique(y_train)) #? i actually have 11 classes but only 6 in the dataset

    # run multiple times with different optimiser/learningrate/architecture
    model, history = train(input_dim, hidden_dim, output_dim, epochs=100, patience=5)