from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader


class TitanicMLP(nn.Module):
    """
    Simple two-layer network for the Titanic dataset.
    """

    def __init__(self, input_dim, hidden_dim, batch_size, learning_rate, epochs):
        super(TitanicMLP, self).__init__()

        # Parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Define the forward pass layers
        self.forward_pass = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        # Define loss function and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        """
        Perform the forward pass.
        """
        return self.forward_pass(x)

    def fit(self, X, y):
        """
        Train the model.
        """
        self.train()

        # Create tensors
        X_tensor = torch.Tensor(X).float()
        Y_tensor = torch.Tensor(y).float()

        # Create DataLoader
        train_dataset = TensorDataset(X_tensor, Y_tensor)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Training loop
        for epoch in range(self.epochs):
            for batch_idx, (features, target) in enumerate(train_loader):
                self.optimizer.zero_grad()  # reset gradients
                outputs = self.forward(features)  # forward pass
                loss = self.criterion(
                    torch.squeeze(outputs), torch.squeeze(target)
                )  # calculate loss
                loss.backward()  # backpropagation
                self.optimizer.step()  # update weights

            # Print progress
            if (epoch + 1) % 10 == 0 and epoch != 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}")

    def predict(self, X):
        """
        Predict the class of the input data X.
        """
        self.eval()  # switch to evaluation mode
        with torch.no_grad():
            X_tensor = torch.Tensor(X)
            y_pred = torch.sigmoid(
                self.forward(X_tensor)
            )  # apply sigmoid for binary output
            y_pred = (
                torch.round(y_pred).squeeze().numpy()
            )  # round to nearest integer (0 or 1) and convert to numpy array
        return y_pred
