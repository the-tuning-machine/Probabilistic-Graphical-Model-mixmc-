import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
from src.estimate.model import DiffusionTransformer


class ThetaYDataset(Dataset):
    """PyTorch Dataset for (y, theta) pairs."""

    def __init__(self, y_samples, theta_params):
        """
        Args:
            y_samples: (n_sample, n_data_point, obs_dim) numpy array
            theta_params: (n_sample, n_data_point, param_dim) numpy array
        """
        self.y_samples = torch.FloatTensor(y_samples)
        self.theta_params = torch.FloatTensor(theta_params)

    def __len__(self):
        return len(self.y_samples)

    def __getitem__(self, idx):
        return self.y_samples[idx], self.theta_params[idx]


class DiffusionTrainer:
    """Trainer for the diffusion transformer model (single-pass training)."""

    def __init__(
        self,
        model: DiffusionTransformer,
        dataset,
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: DiffusionTransformer model
            dataset: DiffusionDataset from data.py
            learning_rate: Learning rate
            batch_size: Batch size
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size

        # Compute normalization statistics for theta parameters
        # Shape: (n_sample, n_data_point, param_dim)
        theta_flat = dataset.theta_params.reshape(-1, dataset.param_dim)  # (n_sample * n_data_point, param_dim)
        self.theta_mean = np.mean(theta_flat, axis=0)  # (param_dim,)
        self.theta_std = np.std(theta_flat, axis=0) + 1e-8  # (param_dim,) with small epsilon to avoid division by zero

        # Normalize theta parameters
        theta_normalized = (dataset.theta_params - self.theta_mean) / self.theta_std

        # Create PyTorch dataset with normalized theta
        full_dataset = ThetaYDataset(
            dataset.y_samples,
            theta_normalized
        )

        # Split into train and test (90/10)
        from torch.utils.data import random_split
        n_total = len(full_dataset)
        n_train = int(0.9 * n_total)
        n_test = n_total - n_train

        self.train_dataset, self.test_dataset = random_split(
            full_dataset,
            [n_train, n_test],
            generator=torch.Generator().manual_seed(42)
        )

        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Loss function (MSE)
        self.criterion = nn.MSELoss()

        # Training history
        self.history = {
            'loss': [],
            'batch': []
        }

    def train_step(self, y_batch, theta_batch):
        """One training step.

        Args:
            y_batch: (batch_size, n_data_point, obs_dim) observations
            theta_batch: (batch_size, n_data_point, param_dim) parameters

        Returns:
            loss: scalar loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move to device
        y_batch = y_batch.to(self.device)
        theta_batch = theta_batch.to(self.device)

        # Sample random timesteps
        t = torch.randint(
            0, self.model.n_timesteps,
            (y_batch.size(0),),
            device=self.device
        ).long()

        # Add noise
        noise = torch.randn_like(theta_batch)
        theta_t, noise = self.model.add_noise(theta_batch, t, noise)

        # Predict noise
        predicted_noise = self.model(theta_t, y_batch, t)

        # Compute loss
        loss = self.criterion(predicted_noise, noise)

        # Backpropagation
        loss.backward()

        # Gradient clipping (more aggressive to prevent instability)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

        self.optimizer.step()

        return loss.item()

    def train(self, verbose: bool = True, log_interval: int = 10):
        """Train the model with a single pass through the dataset.

        Args:
            verbose: Whether to print progress
            log_interval: Interval for logging (in batches)
        """
        if verbose:
            print(f"Training on device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Train dataset size: {len(self.train_dataset)} samples")
            print(f"Test dataset size: {len(self.test_dataset)} samples")
            print(f"Batch size: {self.batch_size}")
            print(f"Total batches: {len(self.train_dataloader)}")
            print()
            print("Starting single-pass training...")
            print()

        total_loss = 0.0
        batch_count = 0

        for batch_idx, (y_batch, theta_batch) in enumerate(self.train_dataloader):
            loss = self.train_step(y_batch, theta_batch)
            total_loss += loss
            batch_count += 1

            self.history['loss'].append(loss)
            self.history['batch'].append(batch_idx)

            if verbose and (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / batch_count
                print(f"Batch {batch_idx + 1}/{len(self.train_dataloader)} - Loss: {loss:.6f} - Avg Loss: {avg_loss:.6f}")

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0

        if verbose:
            print()
            print("Training complete!")
            print(f"Final average loss: {avg_loss:.6f}")

    @torch.no_grad()
    def test_step(self, y_batch, theta_batch):
        """One test step.

        Args:
            y_batch: (batch_size, n_data_point, obs_dim) observations
            theta_batch: (batch_size, n_data_point, param_dim) parameters

        Returns:
            loss: scalar loss value
        """
        self.model.eval()

        # Move to device
        y_batch = y_batch.to(self.device)
        theta_batch = theta_batch.to(self.device)

        # Sample random timesteps
        t = torch.randint(
            0, self.model.n_timesteps,
            (y_batch.size(0),),
            device=self.device
        ).long()

        # Add noise
        noise = torch.randn_like(theta_batch)
        theta_t, noise = self.model.add_noise(theta_batch, t, noise)

        # Predict noise
        predicted_noise = self.model(theta_t, y_batch, t)

        # Compute loss
        loss = self.criterion(predicted_noise, noise)

        return loss.item()

    @torch.no_grad()
    def evaluate_test_set(self, verbose: bool = True):
        """Evaluate the model on the test set.

        Args:
            verbose: Whether to print progress

        Returns:
            Average test loss
        """
        self.model.eval()

        total_loss = 0.0
        batch_count = 0

        for y_batch, theta_batch in self.test_dataloader:
            loss = self.test_step(y_batch, theta_batch)
            total_loss += loss
            batch_count += 1

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0

        if verbose:
            print(f"Test set evaluation - Avg Loss: {avg_loss:.6f}")

        return avg_loss

    @torch.no_grad()
    def evaluate(self, n_samples: int = 10):
        """Evaluate the model by generating predictions from test set.

        Args:
            n_samples: Number of samples to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()

        # Get samples from test dataset
        test_size = len(self.test_dataset)
        n_samples_to_use = min(n_samples, test_size)

        # Collect samples from test set
        y_list = []
        theta_list = []
        for i in range(n_samples_to_use):
            y, theta = self.test_dataset[i]
            y_list.append(y)
            theta_list.append(theta)

        y_eval = torch.stack(y_list).to(self.device)
        theta_true = torch.stack(theta_list)

        # Generate predictions (normalized)
        theta_pred_normalized = self.model.generate(y_eval, device=self.device)
        theta_pred_normalized = theta_pred_normalized.cpu()

        # Denormalize predictions
        theta_mean_torch = torch.FloatTensor(self.theta_mean)
        theta_std_torch = torch.FloatTensor(self.theta_std)
        theta_pred = theta_pred_normalized * theta_std_torch + theta_mean_torch

        # Denormalize true values for fair comparison
        theta_true_denorm = theta_true * theta_std_torch + theta_mean_torch

        # Compute MSE on denormalized values
        mse = torch.mean((theta_pred - theta_true_denorm) ** 2).item()

        return {
            'mse': mse,
            'predictions': theta_pred.numpy(),
            'true_params': theta_true_denorm.numpy(),
            'observations': y_eval.cpu().numpy()
        }

    def save_model(self, filepath: str):
        """Save model checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'obs_dim': self.model.obs_dim,
                'param_dim': self.model.param_dim,
                'd_model': self.model.d_model,
                'n_timesteps': self.model.n_timesteps
            },
            'normalization': {
                'theta_mean': self.theta_mean,
                'theta_std': self.theta_std
            }
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model checkpoint.

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']

        # Load normalization statistics if available
        if 'normalization' in checkpoint:
            self.theta_mean = checkpoint['normalization']['theta_mean']
            self.theta_std = checkpoint['normalization']['theta_std']

        print(f"Model loaded from {filepath}")


def visualize_predictions(
    eval_results: dict,
    n_display: int = 5,
    save_path: Optional[str] = None
):
    """Visualize model predictions.

    Args:
        eval_results: Evaluation results dictionary
        n_display: Number of examples to display
        save_path: Path to save figure (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    predictions = eval_results['predictions'][:n_display]  # (n_display, n_data_point, param_dim)
    true_params = eval_results['true_params'][:n_display]
    observations = eval_results['observations'][:n_display]  # (n_display, n_data_point, obs_dim)

    n_samples = min(n_display, len(predictions))
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4 * n_samples))

    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        ax = axes[i]

        n_data_point = predictions[i].shape[0]
        obs_dim = observations[i].shape[1]

        # Extract means from parameters (first obs_dim dimensions)
        pred_means = predictions[i][:, :obs_dim]  # (n_data_point, obs_dim)
        true_means = true_params[i][:, :obs_dim]  # (n_data_point, obs_dim)
        obs = observations[i]  # (n_data_point, obs_dim)

        # Plot for each dimension
        x = np.arange(n_data_point)
        for dim in range(obs_dim):
            ax.plot(x, pred_means[:, dim], 'o-', label=f'Predicted mean (dim {dim})', alpha=0.7)
            ax.plot(x, true_means[:, dim], 's-', label=f'True mean (dim {dim})', alpha=0.7)
            ax.scatter(x, obs[:, dim], marker='x', s=100, label=f'Observation (dim {dim})', alpha=0.5)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Data point index')
        ax.set_ylabel('Value')
        ax.set_title(f'Sample {i + 1}: {n_data_point} data points')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()
