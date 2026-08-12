import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evoaae_model import Encoder, Decoder


# ============================================================
# EvoAAE Training on WADI
# ============================================================
# Final training configuration used in the project:
#
# Input:
#   RAW + Min-Max normalized WADI windows
#   Shape: (N, 100, 123)
#
# EvoAAE:
#   Encoder: 5 Conv1D layers
#   Filters: 32
#   Kernel size: 2
#   Activation: Sigmoid
#   Latent dimension: 64
#
# Training:
#   Batch size: 1024
#   Epochs: 50
#   Optimizer: Adam
#   Initial learning rate: 0.001
#   Adversarial weight: 0.2
#
# The best model is selected using the lowest average
# training reconstruction loss.
# ============================================================


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TRAIN_NPY = "processed_training_data_raw.npy"

SAVE_DIR = "saved_models_raw"

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Load training data using memory mapping
# ------------------------------------------------------------

print(
    "Opening training data as memmap..."
)

train_mm = np.load(
    TRAIN_NPY,
    mmap_mode="r"
)

print(
    "Memmap shape:",
    train_mm.shape,
    "dtype:",
    train_mm.dtype
)


N_SAMPLES = train_mm.shape[0]
WIN_LEN = train_mm.shape[1]
N_FEATURES = train_mm.shape[2]

TIME_LEN = WIN_LEN
IN_CHANNELS = N_FEATURES


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class NumpyMemmapDataset(Dataset):

    def __init__(self, memmap_array):

        self.arr = memmap_array

    def __len__(self):

        return self.arr.shape[0]

    def __getitem__(self, idx):

        # Original shape:
        # (time, features)

        x = self.arr[idx]

        # Convert to float32

        x = x.astype(
            np.float32
        )

        x = torch.from_numpy(
            x
        )

        # Convert:
        # (100, 123)
        #
        # to:
        # (123, 100)

        x = x.permute(
            1,
            0
        ).contiguous()

        return x


train_dataset = NumpyMemmapDataset(
    train_mm
)


# ------------------------------------------------------------
# Final training hyperparameters
# ------------------------------------------------------------

BATCH_SIZE = 1024

NUM_EPOCHS = 50

LR = 0.001

LAMBDA_ADV = 0.2

LATENT_DIM = 64

SEED = 42


# Reproducibility

torch.manual_seed(SEED)
np.random.seed(SEED)

if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


print(
    f"Total training samples: "
    f"{len(train_dataset)}"
)

print(
    f"Batch size: {BATCH_SIZE}"
)


# ------------------------------------------------------------
# Discriminator for latent space
# ------------------------------------------------------------

class DiscriminatorZ(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                LATENT_DIM,
                128
            ),

            nn.LeakyReLU(
                0.2
            ),

            nn.Linear(
                128,
                64
            ),

            nn.LeakyReLU(
                0.2
            ),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, z):

        return self.net(
            z
        ).view(-1)


# ------------------------------------------------------------
# Discriminator for data space
# ------------------------------------------------------------

class DiscriminatorX(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                IN_CHANNELS,
                64,
                3,
                padding=1
            ),

            nn.LeakyReLU(
                0.2
            ),

            nn.Conv1d(
                64,
                128,
                3,
                padding=1
            ),

            nn.LeakyReLU(
                0.2
            )
        )

        # Determine flattened dimension automatically.

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                IN_CHANNELS,
                TIME_LEN
            )

            h = self.features(
                dummy
            )

            if h.size(-1) > TIME_LEN:

                h = h[
                    ...,
                    :TIME_LEN
                ]

            elif h.size(-1) < TIME_LEN:

                diff = (
                    TIME_LEN
                    - h.size(-1)
                )

                h = nn.functional.pad(
                    h,
                    (0, diff)
                )

            flat_dim = (
                h.shape[1]
                * h.shape[2]
            )

        self.fc1 = nn.Linear(
            flat_dim,
            256
        )

        self.act = nn.LeakyReLU(
            0.2
        )

        self.fc2 = nn.Linear(
            256,
            1
        )

    def forward(self, x):

        h = self.features(
            x
        )

        if h.size(-1) > TIME_LEN:

            h = h[
                ...,
                :TIME_LEN
            ]

        elif h.size(-1) < TIME_LEN:

            diff = (
                TIME_LEN
                - h.size(-1)
            )

            h = nn.functional.pad(
                h,
                (0, diff)
            )

        h = h.view(
            h.size(0),
            -1
        )

        h = self.fc1(
            h
        )

        h = self.act(
            h
        )

        out = self.fc2(
            h
        )

        return out.view(-1)


# ------------------------------------------------------------
# Weight initialization
# ------------------------------------------------------------

def weights_init(m):

    if isinstance(
        m,
        (nn.Conv1d, nn.Linear)
    ):

        nn.init.kaiming_normal_(
            m.weight,
            nonlinearity="relu"
        )

        if m.bias is not None:

            nn.init.zeros_(
                m.bias
            )


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

def make_optimizer(
    params,
    lr
):

    return optim.Adam(
        params,
        lr=lr,
        betas=(0.5, 0.9)
    )


# ------------------------------------------------------------
# Create models
# ------------------------------------------------------------

encoder = Encoder(
    in_channels=IN_CHANNELS,
    time_len=TIME_LEN,
    n_layers=5,
    kernels=32,
    kernel_size=2,
    latent_dim=LATENT_DIM
).to(DEVICE)


decoder = Decoder(
    out_channels=IN_CHANNELS,
    time_len=TIME_LEN,
    kernels=32,
    kernel_size=2,
    latent_dim=LATENT_DIM
).to(DEVICE)


D_z = DiscriminatorZ().to(
    DEVICE
)

D_x = DiscriminatorX().to(
    DEVICE
)


# ------------------------------------------------------------
# Initialize weights
# ------------------------------------------------------------

encoder.apply(
    weights_init
)

decoder.apply(
    weights_init
)

D_z.apply(
    weights_init
)

D_x.apply(
    weights_init
)


# ------------------------------------------------------------
# Optimizers
# ------------------------------------------------------------

opt_ae = make_optimizer(
    list(encoder.parameters())
    + list(decoder.parameters()),
    lr=LR
)

opt_dz = make_optimizer(
    D_z.parameters(),
    lr=LR * 0.5
)

opt_dx = make_optimizer(
    D_x.parameters(),
    lr=LR * 0.5
)


# ------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------

mse = nn.MSELoss()

bce = nn.BCEWithLogitsLoss()


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

print(
    "Starting EvoAAE adversarial "
    "training (RAW data) ..."
)


t0 = time.time()

best_train_recon = float(
    "inf"
)

best_epoch = 0


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    encoder.train()
    decoder.train()
    D_z.train()
    D_x.train()

    running_recon = 0.0

    seen = 0

    pbar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{NUM_EPOCHS}",
        leave=False
    )

    for batch in pbar:

        x_real = batch.to(
            DEVICE
        )

        B = x_real.size(
            0
        )

        real_labels = torch.ones(
            B,
            device=DEVICE
        )

        fake_labels = torch.zeros(
            B,
            device=DEVICE
        )


        # ====================================================
        # 1. Train latent discriminator Dz
        # ====================================================

        with torch.no_grad():

            z_fake = encoder(
                x_real
            )

        z_real = torch.randn(
            B,
            LATENT_DIM,
            device=DEVICE
        )

        dz_real_logits = D_z(
            z_real
        )

        dz_fake_logits = D_z(
            z_fake.detach()
        )

        loss_dz_real = bce(
            dz_real_logits,
            real_labels
        )

        loss_dz_fake = bce(
            dz_fake_logits,
            fake_labels
        )

        loss_dz = (
            loss_dz_real
            + loss_dz_fake
        )

        opt_dz.zero_grad()

        loss_dz.backward()

        opt_dz.step()


        # ====================================================
        # 2. Train data discriminator Dx
        # ====================================================

        with torch.no_grad():

            z_enc_for_dx = encoder(
                x_real
            )

            x_fake = decoder(
                z_enc_for_dx
            )

        dx_real_logits = D_x(
            x_real
        )

        dx_fake_logits = D_x(
            x_fake.detach()
        )

        loss_dx_real = bce(
            dx_real_logits,
            real_labels
        )

        loss_dx_fake = bce(
            dx_fake_logits,
            fake_labels
        )

        loss_dx = (
            loss_dx_real
            + loss_dx_fake
        )

        opt_dx.zero_grad()

        loss_dx.backward()

        opt_dx.step()


        # ====================================================
        # 3. Train Encoder + Decoder
        # ====================================================

        z_enc = encoder(
            x_real
        )

        x_recon = decoder(
            z_enc
        )

        recon_loss = mse(
            x_recon,
            x_real
        )

        dz_fake_logits_for_ae = D_z(
            z_enc
        )

        dx_fake_logits_for_ae = D_x(
            x_recon
        )

        adv_z = bce(
            dz_fake_logits_for_ae,
            real_labels
        )

        adv_x = bce(
            dx_fake_logits_for_ae,
            real_labels
        )

        ae_loss = (
            recon_loss
            + LAMBDA_ADV
            * (adv_z + adv_x)
        )

        opt_ae.zero_grad()

        ae_loss.backward()

        opt_ae.step()


        running_recon += (
            recon_loss.item()
            * B
        )

        seen += B


        pbar.set_postfix({

            "recon":
                f"{recon_loss.item():.6f}",

            "ae_loss":
                f"{ae_loss.item():.6f}",

            "loss_dz":
                f"{loss_dz.item():.4f}",

            "loss_dx":
                f"{loss_dx.item():.4f}"
        })


    # ========================================================
    # Epoch summary
    # ========================================================

    avg_recon = (
        running_recon / seen
        if seen > 0
        else 0.0
    )

    elapsed = (
        time.time() - t0
    ) / 60

    print(
        f"Epoch {epoch}/{NUM_EPOCHS} | "
        f"Train recon avg: {avg_recon:.6f} | "
        f"elapsed {elapsed:.2f} min"
    )


    # ========================================================
    # Learning-rate decay
    # ========================================================

    if epoch % 20 == 0:

        for g in opt_ae.param_groups:
            g["lr"] *= 0.5

        for g in opt_dz.param_groups:
            g["lr"] *= 0.5

        for g in opt_dx.param_groups:
            g["lr"] *= 0.5

        print(
            f"LR decayed at epoch {epoch}"
        )


    # ========================================================
    # Save best model
    # ========================================================

    if avg_recon < best_train_recon:

        best_train_recon = avg_recon

        best_epoch = epoch

        torch.save(
            encoder.state_dict(),
            os.path.join(
                SAVE_DIR,
                "encoder_best.pth"
            )
        )

        torch.save(
            decoder.state_dict(),
            os.path.join(
                SAVE_DIR,
                "decoder_best.pth"
            )
        )

        print(
            f"*** New best model at epoch "
            f"{epoch} with Train recon "
            f"{avg_recon:.6f}"
        )


print(
    f"Training finished. "
    f"Best epoch: {best_epoch}, "
    f"Best Train recon: "
    f"{best_train_recon:.6f}"
)


# ------------------------------------------------------------
# Compute reconstruction errors on full training set
# ------------------------------------------------------------

print(
    "\nComputing reconstruction errors "
    "on full training set using best model..."
)


encoder.load_state_dict(
    torch.load(
        os.path.join(
            SAVE_DIR,
            "encoder_best.pth"
        ),
        map_location=DEVICE
    )
)

decoder.load_state_dict(
    torch.load(
        os.path.join(
            SAVE_DIR,
            "decoder_best.pth"
        ),
        map_location=DEVICE
    )
)


encoder.eval()
decoder.eval()


recon_errors = np.zeros(
    len(train_dataset),
    dtype=np.float32
)


eval_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


idx = 0


with torch.no_grad():

    for batch in tqdm(
        eval_loader,
        desc="Computing recon errors"
    ):

        x = batch.to(
            DEVICE
        )

        z = encoder(
            x
        )

        xhat = decoder(
            z
        )

        errs = torch.mean(
            (x - xhat) ** 2,
            dim=(1, 2)
        ).cpu().numpy()

        recon_errors[
            idx:idx + len(errs)
        ] = errs

        idx += len(errs)


out_path = os.path.join(
    SAVE_DIR,
    "reconstruction_errors_raw.npy"
)


np.save(
    out_path,
    recon_errors
)


print(
    "Saved reconstruction errors to:",
    out_path
)

print(
    "Training complete."
)
