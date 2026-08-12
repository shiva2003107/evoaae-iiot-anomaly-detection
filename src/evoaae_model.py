import torch
import torch.nn as nn


# ============================================================
# Final EvoAAE Model
# ============================================================
# Architecture used in the final reported experiment:
# - Input features : 123
# - Time window    : 100
# - CNN layers     : 5
# - Filters        : 32
# - Kernel size    : 2
# - Activation     : Sigmoid
# - Latent dimension: 64
#
# The model is used as the encoder-decoder component of EvoAAE.
# ============================================================


N_LAYERS = 5
KERNEL_SIZE = 2
KERNELS = 32
LATENT_DIM = 64


class Encoder(nn.Module):
    """
    1D-CNN encoder used in the final EvoAAE architecture.
    """

    def __init__(
        self,
        in_channels=123,
        time_len=100,
        n_layers=N_LAYERS,
        kernels=KERNELS,
        kernel_size=KERNEL_SIZE,
        latent_dim=LATENT_DIM,
    ):
        super().__init__()

        layers = []
        in_ch = in_channels

        for _ in range(n_layers):
            layers.append(
                nn.Conv1d(
                    in_ch,
                    kernels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.Sigmoid())
            in_ch = kernels

        self.conv = nn.Sequential(*layers)

        # Determine the flattened convolutional size automatically.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, time_len)
            h = self.conv(dummy)
            flat_dim = h.shape[1] * h.shape[2]

        self.fc = nn.Linear(flat_dim, latent_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class Decoder(nn.Module):
    """
    1D-CNN decoder used in the final EvoAAE architecture.
    """

    def __init__(
        self,
        out_channels=123,
        time_len=100,
        kernels=KERNELS,
        kernel_size=KERNEL_SIZE,
        latent_dim=LATENT_DIM,
    ):
        super().__init__()

        self.hidden = kernels
        self.time_len = time_len

        self.fc = nn.Linear(
            latent_dim,
            self.hidden * time_len,
        )

        self.deconv = nn.Sequential(
            nn.Conv1d(
                self.hidden,
                self.hidden,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.Sigmoid(),

            nn.Conv1d(
                self.hidden,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)

        x = x.view(
            z.size(0),
            self.hidden,
            self.time_len,
        )

        x = self.deconv(x)

        # Ensure the reconstructed sequence has exactly
        # the required time-window length.
        t = x.size(-1)

        if t > self.time_len:
            x = x[..., :self.time_len]

        elif t < self.time_len:
            x = nn.functional.pad(
                x,
                (0, self.time_len - t),
            )

        return x


if __name__ == "__main__":
    # Simple architecture check.
    # This does not train the model.

    encoder = Encoder()
    decoder = Decoder()

    sample = torch.randn(2, 123, 100)

    latent = encoder(sample)
    reconstructed = decoder(latent)

    print("Input shape       :", sample.shape)
    print("Latent shape      :", latent.shape)
    print("Reconstructed shape:", reconstructed.shape)
