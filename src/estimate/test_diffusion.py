import torch
import numpy as np
from src.estimate.model import DiffusionTransformer

def test_forward_backward_consistency():
    model = DiffusionTransformer(
        obs_dim=1,
        param_dim=2,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        n_timesteps=100
    )
    model.eval()

    batch_size = 4
    n_data_point = 10

    y = torch.randn(batch_size, n_data_point, 1)
    theta_0 = torch.randn(batch_size, n_data_point, 2)

    print("DIFFUSION PROCESS TEST")
    print("Initial theta_0 statistics:")
    print(f"  Mean: {theta_0.mean().item():.4f}")
    print(f"  Std: {theta_0.std().item():.4f}")
    print(f"  Min: {theta_0.min().item():.4f}")
    print(f"  Max: {theta_0.max().item():.4f}")
    print()

    t = torch.tensor([50, 50, 50, 50])
    theta_t, noise = model.add_noise(theta_0, t)

    print("After adding noise at t=50:")
    print(f"  Mean: {theta_t.mean().item():.4f}")
    print(f"  Std: {theta_t.std().item():.4f}")
    print(f"  Min: {theta_t.min().item():.4f}")
    print(f"  Max: {theta_t.max().item():.4f}")
    print()

    with torch.no_grad():
        theta_start = torch.randn(batch_size, n_data_point, 2)
        print("Starting from pure noise:")
        print(f"  Mean: {theta_start.mean().item():.4f}")
        print(f"  Std: {theta_start.std().item():.4f}")
        print()

        theta_t = theta_start.clone()
        for step in [99, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0]:
            if step < 100:
                t = torch.full((batch_size,), step, dtype=torch.long)

                predicted_noise = model.forward(theta_t, y, t)

                alpha_t = model.alphas[step]
                alpha_bar_t = model.alphas_cumprod[step]
                beta_t = model.betas[step]

                coef1 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
                mean = (1.0 / torch.sqrt(alpha_t)) * (theta_t - coef1 * predicted_noise)

                if step > 0:
                    noise_sample = torch.randn_like(theta_t)
                    sigma_t = torch.sqrt(beta_t)
                    theta_t = mean + sigma_t * noise_sample
                else:
                    theta_t = mean

                theta_t = torch.clamp(theta_t, -10.0, 10.0)

                print(f"  t={step:3d} - Mean: {theta_t.mean().item():7.4f}, Std: {theta_t.std().item():7.4f}, Min: {theta_t.min().item():7.4f}, Max: {theta_t.max().item():7.4f}")

        print()
        print("Final output:")
        print(f"  Mean: {theta_t.mean().item():.4f}")
        print(f"  Std: {theta_t.std().item():.4f}")
        print(f"  Min: {theta_t.min().item():.4f}")
        print(f"  Max: {theta_t.max().item():.4f}")

        if theta_t.abs().max() > 100:
            print("\n[FAILED] DIVERGENCE DETECTED! Values are exploding.")
        else:
            print("\n[OK] No divergence detected. Values are stable.")

    print()
    print("TEST COMPLETE")

if __name__ == "__main__":
    test_forward_backward_consistency()
