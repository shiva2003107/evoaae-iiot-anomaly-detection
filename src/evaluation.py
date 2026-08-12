import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score
)


# ============================================================
# Device and paths
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAVE_DIR = "saved_models_raw"


# ============================================================
# Load test data
# ============================================================

X_test = np.load(
    "processed_test_data_raw.npy"
)

y_test = np.load(
    "test_labels_raw.npy"
)


X_test_torch = torch.from_numpy(
    X_test
).float().permute(
    0, 2, 1
)


TIME_LEN = X_test_torch.shape[2]

IN_CHANNELS = X_test_torch.shape[1]


# ============================================================
# EvoAAE configuration
# ============================================================

N_LAYERS = 5

KERNEL_SIZE = 2

KERNELS = 32

LATENT_DIM = 64

activation_fn = nn.Sigmoid


# ============================================================
# Encoder
# ============================================================

class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        layers = []

        in_ch = IN_CHANNELS

        for _ in range(N_LAYERS):

            layers.append(
                nn.Conv1d(
                    in_ch,
                    KERNELS,
                    KERNEL_SIZE,
                    padding=KERNEL_SIZE // 2
                )
            )

            layers.append(
                activation_fn()
            )

            in_ch = KERNELS

        self.conv = nn.Sequential(
            *layers
        )

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                IN_CHANNELS,
                TIME_LEN
            )

            h = self.conv(
                dummy
            )

            flat = (
                h.shape[1]
                * h.shape[2]
            )

        self.fc = nn.Linear(
            flat,
            LATENT_DIM
        )

    def forward(self, x):

        h = self.conv(
            x
        )

        h = h.view(
            h.size(0),
            -1
        )

        return self.fc(
            h
        )


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden = KERNELS

        self.fc = nn.Linear(
            LATENT_DIM,
            self.hidden * TIME_LEN
        )

        self.deconv = nn.Sequential(

            nn.Conv1d(
                self.hidden,
                self.hidden,
                KERNEL_SIZE,
                padding=KERNEL_SIZE // 2
            ),

            activation_fn(),

            nn.Conv1d(
                self.hidden,
                IN_CHANNELS,
                KERNEL_SIZE,
                padding=KERNEL_SIZE // 2
            ),

            nn.Sigmoid()
        )

    def forward(self, z):

        x = self.fc(
            z
        ).view(
            z.size(0),
            self.hidden,
            TIME_LEN
        )

        x = self.deconv(
            x
        )

        # Enforce exact time length

        t = x.size(-1)

        if t > TIME_LEN:

            x = x[
                ...,
                :TIME_LEN
            ]

        elif t < TIME_LEN:

            x = nn.functional.pad(
                x,
                (0, TIME_LEN - t)
            )

        return x


# ============================================================
# Load trained EvoAAE models
# ============================================================

print(
    "\nBuilding EvoAAE encoder/decoder for detection..."
)

encoder = Encoder().to(
    DEVICE
)

decoder = Decoder().to(
    DEVICE
)


print(
    "Loading trained EvoAAE weights..."
)

encoder.load_state_dict(
    torch.load(
        f"{SAVE_DIR}/encoder_best.pth",
        map_location=DEVICE
    )
)

decoder.load_state_dict(
    torch.load(
        f"{SAVE_DIR}/decoder_best.pth",
        map_location=DEVICE
    )
)


encoder.eval()

decoder.eval()


# ============================================================
# Load training reconstruction errors
# ============================================================

train_errors = np.load(
    f"{SAVE_DIR}/reconstruction_errors_raw.npy"
)


mu = train_errors.mean()

sigma = train_errors.std()

threshold = mu + 3 * sigma


print(
    "Train μ:",
    mu
)

print(
    "Train σ:",
    sigma
)

print(
    "3σ threshold:",
    threshold
)


# ============================================================
# Compute reconstruction errors on test data
# ============================================================

loader = DataLoader(
    TensorDataset(
        X_test_torch
    ),
    batch_size=512,
    shuffle=False
)


all_errors = []


with torch.no_grad():

    for (batch,) in loader:

        batch = batch.to(
            DEVICE
        )

        z = encoder(
            batch
        )

        xhat = decoder(
            z
        )

        error = torch.mean(
            (batch - xhat) ** 2,
            dim=(1, 2)
        )

        all_errors.append(
            error.cpu().numpy()
        )


test_errors = np.concatenate(
    all_errors
)


# ============================================================
# Unsupervised μ + 3σ prediction
# ============================================================

y_pred_3sigma = (
    test_errors >= threshold
).astype(
    int
)


print(
    "\n=== 3σ THRESHOLD RESULTS ==="
)


print(
    "Confusion matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred_3sigma
    )
)


print(
    "\nClassification report:"
)

print(
    classification_report(
        y_test,
        y_pred_3sigma,
        digits=4
    )
)


print(
    "ROC-AUC (score-based):",
    roc_auc_score(
        y_test,
        test_errors
    )
)


# ============================================================
# Optional: Best-F1 threshold analysis
# ============================================================

best_f1 = -1

best_threshold = None


for threshold_candidate in np.linspace(
    test_errors.min(),
    test_errors.max(),
    200
):

    predictions = (
        test_errors >= threshold_candidate
    ).astype(
        int
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    if f1 > best_f1:

        best_f1 = f1

        best_threshold = (
            threshold_candidate
        )


print(
    "\n=== BEST-F1 THRESHOLD "
    "(analysis only) ==="
)


print(
    "Best threshold:",
    best_threshold
)

print(
    "Best F1:",
    best_f1
)


pred_best = (
    test_errors >= best_threshold
).astype(
    int
)


print(
    "Confusion matrix (best-F1):"
)

print(
    confusion_matrix(
        y_test,
        pred_best
    )
)


print(
    "\nClassification report "
    "(best-F1):"
)

print(
    classification_report(
        y_test,
        pred_best,
        digits=4
    )
)
