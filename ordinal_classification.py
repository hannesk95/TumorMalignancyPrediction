import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Sample Dataset
class SampleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Define the Ordinal Classification Model
class OrdinalClassifier(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(OrdinalClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, n_classes - 1)  # n_classes - 1 thresholds for ordinal classification

    def forward(self, x):
        logits = self.fc(x)
        # Apply sigmoid to each logit to predict the probability of being greater than or equal to the threshold
        probabilities = torch.sigmoid(logits)
        return probabilities

# Ordinal Loss (Cumulative Link Loss)
def ordinal_loss(y_pred, y_true, n_classes):
    y_true = y_true.view(-1, 1)  # Shape (batch_size, 1)
    y_true_one_hot = torch.zeros(y_pred.size(), device=y_pred.device)
    
    # Create a one-hot like encoding where 1 is filled for all classes <= y_true
    for i in range(n_classes - 1):
        y_true_one_hot[:, i] = (y_true > i).float().squeeze()

    loss = F.binary_cross_entropy(y_pred, y_true_one_hot)
    return loss

# Generate some synthetic data for demonstration
def generate_synthetic_data(n_samples=1000, n_features=10, n_classes=5):
    data = torch.randn(n_samples, n_features)
    labels = torch.randint(0, n_classes, (n_samples,))
    return data, labels

# Training function
def train_ordinal_classifier(model, data_loader, optimizer, n_classes, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_data, batch_labels in data_loader:
            optimizer.zero_grad()
            predictions = model(batch_data)
            loss = ordinal_loss(predictions, batch_labels, n_classes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

if __name__ == "__main__":
    # Define hyperparameters
    n_samples = 1000
    n_features = 10
    n_classes = 5  # Ordinal levels, e.g., malignancy levels from 1 to 5
    batch_size = 32
    learning_rate = 0.001
    epochs = 20

    # Generate synthetic dataset
    data, labels = generate_synthetic_data(n_samples=n_samples, n_features=n_features, n_classes=n_classes)
    dataset = SampleDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer
    model = OrdinalClassifier(input_dim=n_features, n_classes=n_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_ordinal_classifier(model, data_loader, optimizer, n_classes=n_classes, epochs=epochs)

    print("Training complete.")
