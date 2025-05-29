import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Tuple, Optional
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogDataset(Dataset):
    """Dataset class for ship logs."""
    def __init__(self, data: np.ndarray):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 10):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ShipLogAutoencoder:
    def __init__(self, input_dim: int, encoding_dim: int = 10, 
                 learning_rate: float = 0.001, batch_size: int = 32,
                 num_epochs: int = 100, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        
        # Initialize model
        self.model = Autoencoder(input_dim, encoding_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # For storing reconstruction errors
        self.threshold = None
        self.mean_error = None
        self.std_error = None

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error for input data."""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            reconstructed = self.model(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error.cpu()

    def fit(self, X: np.ndarray, validation_split: float = 0.1) -> Tuple[list, list]:
        """Train the autoencoder."""
        try:
            # Split data into train and validation
            val_size = int(len(X) * validation_split)
            train_size = len(X) - val_size
            train_data, val_data = torch.utils.data.random_split(
                LogDataset(X), [train_size, val_size]
            )

            train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=self.batch_size)

            train_losses = []
            val_losses = []

            logger.info(f"Starting training on device: {self.device}")
            
            for epoch in range(self.num_epochs):
                # Training
                self.model.train()
                train_loss = 0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    self.optimizer.zero_grad()
                    reconstructed = self.model(batch)
                    loss = self.criterion(reconstructed, batch)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        reconstructed = self.model(batch)
                        loss = self.criterion(reconstructed, batch)
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{self.num_epochs}], "
                              f"Train Loss: {train_loss:.6f}, "
                              f"Val Loss: {val_loss:.6f}")

            # Compute threshold for anomaly detection
            self.model.eval()
            errors = []
            with torch.no_grad():
                for batch in train_loader:
                    batch = batch.to(self.device)
                    reconstructed = self.model(batch)
                    error = torch.mean((batch - reconstructed) ** 2, dim=1)
                    errors.extend(error.cpu().numpy())

            self.mean_error = np.mean(errors)
            self.std_error = np.std(errors)
            # Set threshold as mean + 2 standard deviations
            self.threshold = self.mean_error + 2 * self.std_error

            logger.info(f"Training completed. Threshold set to: {self.threshold:.6f}")
            return train_losses, val_losses

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return [], []

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomaly scores for input data."""
        try:
            self.model.eval()
            dataset = LogDataset(X)
            dataloader = DataLoader(dataset, batch_size=self.batch_size)
            
            reconstruction_errors = []
            anomalies = []
            
            with torch.no_grad():
                for batch in dataloader:
                    batch = batch.to(self.device)
                    reconstructed = self.model(batch)
                    error = torch.mean((batch - reconstructed) ** 2, dim=1)
                    reconstruction_errors.extend(error.cpu().numpy())
            
            reconstruction_errors = np.array(reconstruction_errors)
            anomalies = reconstruction_errors > self.threshold
            
            return reconstruction_errors, anomalies

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return np.array([]), np.array([])

    def save_model(self, path: str):
        """Save model state."""
        try:
            state = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'threshold': self.threshold,
                'mean_error': self.mean_error,
                'std_error': self.std_error,
                'input_dim': self.input_dim,
                'encoding_dim': self.encoding_dim
            }
            torch.save(state, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    @classmethod
    def load_model(cls, path: str) -> Optional['ShipLogAutoencoder']:
        """Load model state."""
        try:
            state = torch.load(path)
            autoencoder = cls(
                input_dim=state['input_dim'],
                encoding_dim=state['encoding_dim']
            )
            autoencoder.model.load_state_dict(state['model_state_dict'])
            autoencoder.optimizer.load_state_dict(state['optimizer_state_dict'])
            autoencoder.threshold = state['threshold']
            autoencoder.mean_error = state['mean_error']
            autoencoder.std_error = state['std_error']
            logger.info(f"Model loaded from {path}")
            return autoencoder
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

if __name__ == "__main__":
    # Example usage
    input_dim = 20  # Example dimension
    X = np.random.randn(1000, input_dim)  # Example data
    
    # Initialize and train model
    autoencoder = ShipLogAutoencoder(input_dim=input_dim)
    train_losses, val_losses = autoencoder.fit(X)
    
    # Make predictions
    scores, anomalies = autoencoder.predict(X)
    print(f"Detected {sum(anomalies)} anomalies")
    
    # Save and load test
    autoencoder.save_model('autoencoder_model.pth')
    loaded_model = ShipLogAutoencoder.load_model('autoencoder_model.pth')
    if loaded_model:
        new_scores, new_anomalies = loaded_model.predict(X)
        print(f"Loaded model detected {sum(new_anomalies)} anomalies")
