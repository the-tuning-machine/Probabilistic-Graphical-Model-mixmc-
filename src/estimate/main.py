import numpy as np
import torch
from src.estimate.distribution import NormalDistribution
from src.estimate.data import DiffusionDataset
from src.estimate.model import DiffusionTransformer
from src.estimate.training import DiffusionTrainer, visualize_predictions


def main():
    """Main function to train the diffusion transformer model."""

    # ========== PARAMETERS ==========
    # Dirichlet Process parameters
    alpha = 0.01  # Concentration parameter
    n_sample = 1000000  # Number of samples in dataset
    n_data_point = 20  # Number of (theta, y) pairs per sample
    obs_dim = 1  # Observation space dimension

    # Base distribution G_0 (multivariate Gaussian)
    base_mean = np.zeros(obs_dim)
    base_cov = np.eye(obs_dim)
    G_0 = NormalDistribution(mean=base_mean, covariance=base_cov)

    # Diffusion model parameters
    d_model = 8  # Hidden dimension
    num_heads = 2  # Number of attention heads
    num_layers = 16  # Number of transformer layers
    n_timesteps = 1000  # Number of diffusion steps (reduced for stability)
    dropout = 0

    # Training parameters
    learning_rate = 1e-3
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ========== DATA GENERATION ==========
    print("=" * 60)
    print("DATA GENERATION")
    print("=" * 60)
    print(f"Dirichlet Process (Chinese Restaurant Process):")
    print(f"  - alpha = {alpha}")
    print(f"  - Number of samples: {n_sample}")
    print(f"  - Number of data points per sample: {n_data_point}")
    print(f"  - Base distribution G_0: Gaussian N({base_mean}, Sigma)")
    print(f"  - Observation dimension: {obs_dim}")
    print()

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create dataset using Chinese Restaurant Process
    print("Generating dataset...")
    dataset = DiffusionDataset(
        n_sample=n_sample,
        n_data_point=n_data_point,
        alpha=alpha,
        G_0=G_0
    )

    print(f"\nDataset created:")
    print(f"  - Observations y: {dataset.y_samples.shape}")
    print(f"  - Parameters theta: {dataset.theta_params.shape}")
    print(f"  - Observation dimension: {dataset.obs_dim}")
    print(f"  - Parameter dimension: {dataset.param_dim}")
    print()

    # Display statistics
    print("Observation statistics:")
    print(f"  - Mean: {np.mean(dataset.y_samples.reshape(-1, obs_dim), axis=0)}")
    print(f"  - Std: {np.std(dataset.y_samples.reshape(-1, obs_dim), axis=0)}")
    print()

    # ========== MODEL CREATION ==========
    print("=" * 60)
    print("DIFFUSION TRANSFORMER MODEL")
    print("=" * 60)

    model = DiffusionTransformer(
        obs_dim=dataset.obs_dim,
        param_dim=dataset.param_dim,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        n_timesteps=n_timesteps
    )

    print(f"Model created:")
    print(f"  - Input dimension (observations): {dataset.obs_dim}")
    print(f"  - Output dimension (parameters): {dataset.param_dim}")
    print(f"  - Hidden dimension: {d_model}")
    print(f"  - Number of attention heads: {num_heads}")
    print(f"  - Number of transformer layers: {num_layers}")
    print(f"  - Number of diffusion timesteps: {n_timesteps}")
    print(f"  - Device: {device}")
    print()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {n_params:,}")
    print()

    # ========== TRAINING ==========
    print("=" * 60)
    print("TRAINING (SINGLE PASS)")
    print("=" * 60)

    trainer = DiffusionTrainer(
        model=model,
        dataset=dataset,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device
    )

    print(f"Normalization statistics:")
    print(f"  - Theta mean: {trainer.theta_mean}")
    print(f"  - Theta std: {trainer.theta_std}")
    print()

    trainer.train(
        verbose=True,
        log_interval=10
    )

    # ========== TEST SET EVALUATION ==========
    print()
    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    test_loss = trainer.evaluate_test_set(verbose=True)

    # ========== EVALUATION (Generation on Test Samples) ==========
    print()
    print("=" * 60)
    print("GENERATION EVALUATION")
    print("=" * 60)

    eval_results = trainer.evaluate(n_samples=10)

    print(f"MSE (predictions vs true values): {eval_results['mse']:.6f}")
    print()

    # Visualize predictions
    try:
        visualize_predictions(
            eval_results,
            n_display=3,
            save_path="predictions_diffusion.png"
        )
    except Exception as e:
        print(f"Visualization error: {e}")

    # ========== SAVE MODEL ==========
    print()
    print("=" * 60)
    print("SAVE MODEL")
    print("=" * 60)

    trainer.save_model("diffusion_transformer.pt")

    # ========== GENERATION TEST ==========
    print()
    print("=" * 60)
    print("GENERATION TEST")
    print("=" * 60)

    # Take a test sample
    test_idx = 0
    y_test = torch.FloatTensor(dataset.y_samples[test_idx:test_idx+1])  # (1, n_data_point, obs_dim)
    theta_true = dataset.theta_params[test_idx]  # (n_data_point, param_dim)

    print(f"Test sample:")
    print(f"  - Shape: {y_test.shape}")
    print(f"  - First observation: {y_test[0, 0].numpy()}")
    print(f"  - True parameters (first point, mean): {theta_true[0, :obs_dim]}")
    print()

    # Generate prediction
    print("Generating prediction...")
    model.eval()
    with torch.no_grad():
        prediction_normalized = model.sample(y_test, device=device)
        prediction_normalized = prediction_normalized.cpu().numpy()[0]  # (n_data_point, param_dim)

    # Denormalize prediction
    prediction = prediction_normalized * trainer.theta_std + trainer.theta_mean

    print(f"\nPrediction shape: {prediction.shape}")
    print(f"First predicted mean: {prediction[0, :obs_dim]}")
    print(f"First true mean: {theta_true[0, :obs_dim]}")
    print(f"First predicted cov: {prediction[0, obs_dim:]}")
    print(f"First true cov: {theta_true[0, obs_dim:]}")
    print(f"Error on first point: {np.linalg.norm(prediction[0, :obs_dim] - theta_true[0, :obs_dim]):.6f}")
    print()

    # Compute average error across all points
    avg_error = np.mean([np.linalg.norm(prediction[i, :obs_dim] - theta_true[i, :obs_dim])
                         for i in range(n_data_point)])
    print(f"Average error across {n_data_point} points: {avg_error:.6f}")

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
