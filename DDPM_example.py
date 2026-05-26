import argparse
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =========================
# 1. Basic settings
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Train or sample a tiny 16x16 DDPM.")
    parser.add_argument("--skip-train", "--generate-only", action="store_true",
                        help="Load a saved checkpoint and generate without training.")
    parser.add_argument("--checkpoint", default=os.path.join(os.path.dirname(__file__), "checkpoints", "ddpm_16x16.pt"),
                        help="Path to save/load the model checkpoint.")
    parser.add_argument("--T", type=int, default=200, help="Number of diffusion timesteps.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--train-steps", type=int, default=30000, help="Number of training steps.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--sampling-noise-scale", "--generation-std-scale",
                        dest="sampling_noise_scale", type=float, default=0.5,
                        help="Scale for reverse-process sampling noise.")
    parser.add_argument("--initial-noise-std", type=float, default=1.0,
                        help="Standard deviation of the initial pure noise x_T.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed. Ignored when --random-seed is set.")
    parser.add_argument("--random-seed", action="store_true",
                        help="Use a different random seed on each run.")
    return parser.parse_args()


args = parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

T = args.T
image_size = 16
image_channels = 1
image_dim = image_channels * image_size * image_size
batch_size = args.batch_size
train_steps = args.train_steps
lr = args.lr
sampling_noise_scale = args.sampling_noise_scale
initial_noise_std = args.initial_noise_std
do_train = not args.skip_train
checkpoint_path = args.checkpoint
seed = int(time.time_ns() % (2 ** 32)) if args.random_seed else args.seed

torch.manual_seed(seed)
np.random.seed(seed)

print(
    f"config | T={T} | batch_size={batch_size} | train_steps={train_steps} | "
    f"lr={lr} | sampling_noise_scale={sampling_noise_scale} | "
    f"initial_noise_std={initial_noise_std} | seed={seed} | do_train={do_train}"
)
print(f"checkpoint = {checkpoint_path}")


# =========================
# 2. beta, alpha, alpha_bar
# =========================

def make_beta_schedule(T, beta_start=1e-4, beta_end=0.01):
    beta = torch.zeros(T + 1)
    beta[1:] = torch.linspace(beta_start, beta_end, T)

    alpha = torch.ones(T + 1)
    alpha[1:] = 1.0 - beta[1:]

    alpha_bar = torch.ones(T + 1)
    for t in range(1, T + 1):
        alpha_bar[t] = alpha_bar[t - 1] * alpha[t]

    return beta.to(device), alpha.to(device), alpha_bar.to(device)


beta, alpha, alpha_bar = make_beta_schedule(T)


# =========================
# 3. A single fixed 16x16x1 training image x_0
# =========================

def make_fixed_x0_image():
    x0 = torch.full((1, image_channels, image_size, image_size), -1.0)

    # A simple high-contrast grayscale pattern in [-1, 1].
    x0[:, :, 2:14, 2:14] = -0.35
    x0[:, :, 4:12, 4:12] = 0.25
    x0[:, :, 6:10, 6:10] = 0.95
    x0[:, :, 7:9, 2:14] = 1.0
    x0[:, :, 2:14, 7:9] = 1.0

    return x0.to(device)


fixed_x0 = make_fixed_x0_image()


def sample_x0(batch_size):
    return fixed_x0.expand(batch_size, -1, -1, -1).clone()


# =========================
# 4. Forward noising q(x_t | x_0)
# =========================

def q_sample(x0, t, eps):
    """
    x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps

    x0:  shape [B, 1, 16, 16]
    t:   shape [B]
    eps: shape [B, 1, 16, 16]
    """
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)

    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps
    return xt


# =========================
# 5. Timestep embedding
# =========================

def timestep_embedding(t, dim=32):
    """
    Simple sinusoidal timestep embedding.

    t: shape [B]
    return: shape [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device).float() / half
    )

    args = t.float().view(-1, 1) * freqs.view(1, -1)

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


# =========================
# 6. Neural network epsilon_theta(x_t, t)
# =========================

class EpsilonTheta(nn.Module):
    def __init__(self, time_dim=64, hidden_dim=512):
        super().__init__()
        self.time_dim = time_dim

        self.net = nn.Sequential(
            nn.Linear(image_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, image_dim),
        )

    def forward(self, xt, t):
        b = xt.shape[0]
        xt_flat = xt.view(b, -1)
        t_emb = timestep_embedding(t, self.time_dim)
        inp = torch.cat([xt_flat, t_emb], dim=1)
        eps_pred = self.net(inp)
        return eps_pred.view(b, image_channels, image_size, image_size)


model = EpsilonTheta().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# =========================
# 7. Train epsilon prediction on noisy versions of one image
# =========================

loss_history = []

if do_train:
    for step in range(1, train_steps + 1):
        x0 = sample_x0(batch_size)

        # Random timestep t.
        t = torch.randint(1, T + 1, (batch_size,), device=device)

        # True noise epsilon.
        eps = torch.randn_like(x0)

        # Build x_t from the same fixed x_0 and fresh noise.
        xt = q_sample(x0, t, eps)

        # Predict epsilon.
        eps_pred = model(xt, t)

        # DDPM simple loss.
        loss = F.mse_loss(eps_pred, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % 1000 == 0:
            print(f"step {step:5d} | loss = {loss.item():.6f}")

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "config": {
                "T": T,
                "image_size": image_size,
                "image_channels": image_channels,
                "batch_size": batch_size,
                "train_steps": train_steps,
                "lr": lr,
                "sampling_noise_scale": sampling_noise_scale,
                "initial_noise_std": initial_noise_std,
                "seed": seed,
            },
        },
        checkpoint_path,
    )
    print(f"saved checkpoint to {checkpoint_path}")
else:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run without --skip-train first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_history = checkpoint.get("loss_history", [])
    print(f"loaded checkpoint from {checkpoint_path}")

model.eval()


# =========================
# 8. Reverse generation with the trained network
# =========================

@torch.no_grad()
def p_sample(xt, t_int):
    """
    Sample from p_theta(x_{t-1} | x_t).

    mu_theta(x_t,t)
    =
    1/sqrt(alpha_t) *
    (x_t - beta_t / sqrt(1-alpha_bar_t) * eps_theta(x_t,t))
    """
    b = xt.shape[0]
    t = torch.full((b,), t_int, device=device, dtype=torch.long)

    eps_pred = model(xt, t)

    beta_t = beta[t].view(-1, 1, 1, 1)
    alpha_t = alpha[t].view(-1, 1, 1, 1)
    alpha_bar_t = alpha_bar[t].view(-1, 1, 1, 1)

    mean = (1.0 / torch.sqrt(alpha_t)) * (
        xt - beta_t / torch.sqrt(1.0 - alpha_bar_t) * eps_pred
    )

    if t_int > 1:
        # Simplified reverse variance: use beta_t.
        z = torch.randn_like(xt)
        sigma_t = torch.sqrt(beta_t) * sampling_noise_scale
        x_prev = mean + sigma_t * z
    else:
        # No extra noise in the final step.
        x_prev = mean

    return x_prev


@torch.no_grad()
def sample_reverse(num_samples, start_from_noised_x0=False):
    """
    Denoise back to x_0.

    If start_from_noised_x0 is True, start from q(x_T | x_0), which is the
    restoration setting. Otherwise, start from pure N(0, 1) noise.
    """
    if start_from_noised_x0:
        x0 = sample_x0(num_samples)
        t = torch.full((num_samples,), T, device=device, dtype=torch.long)
        xt = q_sample(x0, t, torch.randn_like(x0))
    else:
        xt = (
            torch.randn(num_samples, image_channels, image_size, image_size, device=device)
            * initial_noise_std
        )

    trajectory = []
    trajectory.append(xt.detach().cpu())

    for t_int in range(T, 0, -1):
        xt = p_sample(xt, t_int)
        trajectory.append(xt.detach().cpu())

    return xt.detach().cpu(), trajectory


generated, trajectory = sample_reverse(16, start_from_noised_x0=False)


# =========================
# 9. Visualize results
# =========================

imgs_dir = os.path.join(os.path.dirname(__file__), "imgs")
os.makedirs(imgs_dir, exist_ok=True)


def to_numpy_img(x):
    x = x.detach().cpu().squeeze().clamp(-1.0, 1.0)
    return ((x + 1.0) / 2.0).numpy()


target_img = to_numpy_img(fixed_x0[0])
generated_img = to_numpy_img(generated[0])
abs_error = np.abs(generated_img - target_img)
reconstruction_mse = F.mse_loss(generated[0:1].to(device), fixed_x0).item()
print(f"sample reconstruction mse = {reconstruction_mse:.6f}")

plt.figure(figsize=(3, 3))
plt.imshow(target_img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(os.path.join(imgs_dir, "target_x0_image.png"), dpi=200)
plt.close()


plt.figure(figsize=(8, 3))

plt.subplot(1, 3, 1)
plt.imshow(target_img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
plt.title("target x0")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(generated_img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
plt.title("generated x0")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(abs_error, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
plt.title("abs error")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(imgs_dir, "target_vs_generated_image.png"), dpi=200)
plt.close()


plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(to_numpy_img(generated[i]), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.axis("off")
plt.tight_layout(pad=0.2)
plt.savefig(os.path.join(imgs_dir, "generated_image_samples.png"), dpi=200)
plt.close()


plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.title("Training loss")
plt.xlabel("training step")
plt.ylabel("MSE loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(imgs_dir, "training_loss.png"), dpi=200)
plt.close()


noise_steps = [0, T // 10, T // 4, T // 2, 3 * T // 4, T]
plt.figure(figsize=(10, 2))
for i, t_int in enumerate(noise_steps):
    plt.subplot(1, len(noise_steps), i + 1)
    if t_int == 0:
        noisy = fixed_x0
    else:
        t = torch.full((1,), t_int, device=device, dtype=torch.long)
        noisy = q_sample(fixed_x0, t, torch.randn_like(fixed_x0))
    plt.imshow(to_numpy_img(noisy[0]), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    plt.title(f"t={t_int}")
    plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(imgs_dir, "forward_noising_image.png"), dpi=200)
plt.close()
