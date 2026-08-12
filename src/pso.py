# Binary PSO for EvoAAE Hyperparameter Search

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# 1. Load Preprocessed Training Data (RAW)

print("Loading preprocessed RAW WADI dataset...")

train_np = np.load(
    "processed_training_data_raw.npy",
    mmap_mode="r"
)

print("Loaded array shape:", train_np.shape)


# Convert to torch tensor for CNN:
# (N, 100, 123) -> (N, 123, 100)

train_tensor = torch.from_numpy(train_np).float().permute(0, 2, 1)

full_dataset = TensorDataset(train_tensor)

IN_CHANNELS = train_tensor.shape[1]
TIME_LEN = train_tensor.shape[2]

print(
    f"IN_CHANNELS = {IN_CHANNELS}, "
    f"TIME_LEN = {TIME_LEN}"
)


# 2. Subset for PSO Fitness

SUBSET_SIZE = 3000

if len(full_dataset) > SUBSET_SIZE:
    idx = np.random.choice(
        len(full_dataset),
        SUBSET_SIZE,
        replace=False
    )

    idx = torch.from_numpy(idx).long()
    subset_tensor = train_tensor[idx]

else:
    subset_tensor = train_tensor


dataset_subset = TensorDataset(subset_tensor)

print(
    f"PSO will use {len(dataset_subset)} "
    f"samples for fitness evaluation."
)


# 3. Hyperparameter Search Space

BATCH_SIZE_LIST = [1024, 2048, 4096, 6144]

OPTIMIZER_NAMES = [
    "adamax",
    "adam",
    "rmsprop",
    "adadelta"
]

LR_LIST = [
    0.0001,
    0.0005,
    0.001,
    0.005
]

NUM_LAYERS_LIST = [
    3,
    4,
    5,
    6
]

KERNELS_LIST = [
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256
]

KERNEL_SIZE_LIST = [
    1,
    2,
    3,
    4
]

ACTIVATION_NAMES = [
    "sigmoid",
    "tanh",
    "relu",
    "none"
]


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "none": nn.Identity
}


# Bit allocation (total = 15 bits)

BIT_LEN = 15


# 4. Binary PSO Hyperparameters

POP_SIZE = 20
MAX_GEN = 30

W_MAX = 0.8
W_MIN = 0.4

C1 = 1.5
C2 = 1.5

V_MAX = 10.0
V_MIN = -10.0


# Fitness evaluation limits

MAX_LOCAL_EPOCHS = 1
MAX_BATCHES_PER_EPOCH = 5


# 5. Utility:
# Decode Binary Position -> Hyperparameters

def bits_to_int(bits):
    """Convert list/array of bits (0/1) to integer."""

    val = 0

    for b in bits:
        val = (val << 1) | int(b)

    return val


def decode_position(position_bits):
    """
    Decode binary PSO position into hyperparameters.

    Returns:
    batch_size,
    optimizer,
    learning_rate,
    number_of_layers,
    kernels,
    kernel_size,
    activation
    """

    bits = np.array(position_bits).astype(int)

    # Bit segments

    bits_B = bits[0:2]
    bits_opt = bits[2:4]
    bits_lr = bits[4:6]
    bits_n = bits[6:8]
    bits_oc = bits[8:11]
    bits_ks = bits[11:13]
    bits_af = bits[13:15]

    idx_B = bits_to_int(bits_B)
    idx_opt = bits_to_int(bits_opt)
    idx_lr = bits_to_int(bits_lr)
    idx_n = bits_to_int(bits_n)
    idx_oc = bits_to_int(bits_oc)
    idx_ks = bits_to_int(bits_ks)
    idx_af = bits_to_int(bits_af)

    # Clamp indices into valid ranges

    idx_B = max(
        0,
        min(idx_B, len(BATCH_SIZE_LIST) - 1)
    )

    idx_opt = max(
        0,
        min(idx_opt, len(OPTIMIZER_NAMES) - 1)
    )

    idx_lr = max(
        0,
        min(idx_lr, len(LR_LIST) - 1)
    )

    idx_n = max(
        0,
        min(idx_n, len(NUM_LAYERS_LIST) - 1)
    )

    idx_oc = max(
        0,
        min(idx_oc, len(KERNELS_LIST) - 1)
    )

    idx_ks = max(
        0,
        min(idx_ks, len(KERNEL_SIZE_LIST) - 1)
    )

    idx_af = max(
        0,
        min(idx_af, len(ACTIVATION_NAMES) - 1)
    )

    batch_size = BATCH_SIZE_LIST[idx_B]
    opt_name = OPTIMIZER_NAMES[idx_opt]
    lr = LR_LIST[idx_lr]
    n_layers = NUM_LAYERS_LIST[idx_n]
    kernels = KERNELS_LIST[idx_oc]
    ksize = KERNEL_SIZE_LIST[idx_ks]
    act_name = ACTIVATION_NAMES[idx_af]

    return (
        batch_size,
        opt_name,
        lr,
        n_layers,
        kernels,
        ksize,
        act_name
    )


# 6. Small Conv Autoencoder
# used for PSO fitness evaluation

class SmallConvAE(nn.Module):

    def __init__(
        self,
        n_layers,
        kernel_size,
        kernel_count,
        activation_fn
    ):

        super().__init__()

        enc_layers = []
        in_ch = IN_CHANNELS

        for _ in range(n_layers):

            enc_layers.append(
                nn.Conv1d(
                    in_ch,
                    kernel_count,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2
                )
            )

            enc_layers.append(
                activation_fn()
            )

            in_ch = kernel_count

        self.encoder = nn.Sequential(
            *enc_layers
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                kernel_count,
                IN_CHANNELS,
                kernel_size,
                stride=1,
                padding=kernel_size // 2
            ),
            nn.Sigmoid()
        )

    def forward(self, x):

        z = self.encoder(x)

        out = self.decoder(z)

        # Adjust time length if mismatch

        if out.size(-1) > x.size(-1):

            out = out[:, :, :x.size(-1)]

        elif out.size(-1) < x.size(-1):

            diff = x.size(-1) - out.size(-1)

            out = nn.functional.pad(
                out,
                (0, diff)
            )

        return out


# 7. Fitness Function

def fitness(position_bits, dataset_subset):

    """
    1. Decode bits into hyperparameters.
    2. Train a small convolutional autoencoder.
    3. Return average MSE as fitness.
    """

    (
        batch_size,
        opt_name,
        lr,
        n_layers,
        kernels,
        ksize,
        act_name
    ) = decode_position(position_bits)

    activation_fn = ACTIVATION_MAP[act_name]

    model = SmallConvAE(
        n_layers,
        ksize,
        kernels,
        activation_fn
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer_dict = {
        "adam": optim.Adam,
        "adamax": optim.Adamax,
        "rmsprop": optim.RMSprop,
        "adadelta": optim.Adadelta
    }

    opt_class = optimizer_dict[opt_name]

    optim_model = opt_class(
        model.parameters(),
        lr=lr
    )

    loader = DataLoader(
        dataset_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    model.train()

    total_loss = 0.0
    total_count = 0

    for epoch in range(MAX_LOCAL_EPOCHS):

        batch_count = 0

        for (x_batch,) in loader:

            x_batch = x_batch.to(DEVICE)

            optim_model.zero_grad()

            x_recon = model(x_batch)

            loss = criterion(
                x_recon,
                x_batch
            )

            loss.backward()

            optim_model.step()

            bs_here = x_batch.size(0)

            total_loss += (
                loss.item() * bs_here
            )

            total_count += bs_here

            batch_count += 1

            if batch_count >= MAX_BATCHES_PER_EPOCH:
                break

    if total_count == 0:

        avg_mse = 1e9

    else:

        avg_mse = (
            total_loss / total_count
        )

    return (
        avg_mse,
        (
            batch_size,
            opt_name,
            lr,
            n_layers,
            kernels,
            ksize,
            act_name
        )
    )


# 8. Initialize Binary PSO Population

def sigmoid(x):

    return 1.0 / (
        1.0 + np.exp(-x)
    )


particles_pos = np.random.uniform(
    low=-1.0,
    high=1.0,
    size=(POP_SIZE, BIT_LEN)
)

particles_vel = np.zeros(
    (POP_SIZE, BIT_LEN),
    dtype=float
)

pbest_pos = particles_pos.copy()

pbest_fit = np.full(
    POP_SIZE,
    np.inf,
    dtype=float
)

gbest_pos = None
gbest_fit = np.inf
gbest_cfg = None


# 9. PSO Main Loop

for gen in range(
    1,
    MAX_GEN + 1
):

    print(
        f"\nGeneration {gen}/{MAX_GEN}"
    )

    if MAX_GEN > 1:

        w = (
            W_MAX
            + (
                gen
                * (W_MAX - W_MIN)
            )
            / MAX_GEN
        )

    else:

        w = W_MAX

    # Evaluate each particle

    for i in range(POP_SIZE):

        probs = sigmoid(
            particles_pos[i]
        )

        bits = (
            probs >= 0.5
        ).astype(int)

        fit_val, decoded_cfg = fitness(
            bits,
            dataset_subset
        )

        # Personal best

        if fit_val < pbest_fit[i]:

            pbest_fit[i] = fit_val

            pbest_pos[i] = (
                particles_pos[i].copy()
            )

        # Global best

        if fit_val < gbest_fit:

            gbest_fit = fit_val

            gbest_pos = (
                particles_pos[i].copy()
            )

            gbest_cfg = decoded_cfg

            print(
                f"New global best fitness: "
                f"{gbest_fit:.6f}"
            )

            print(
                f"Best config so far: "
                f"batch={gbest_cfg[0]}, "
                f"opt={gbest_cfg[1]}, "
                f"lr={gbest_cfg[2]}, "
                f"layers={gbest_cfg[3]}, "
                f"kernels={gbest_cfg[4]}, "
                f"ksize={gbest_cfg[5]}, "
                f"act={gbest_cfg[6]}"
            )

    print(
        f"Best fitness so far: "
        f"{gbest_fit:.6f}"
    )

    # Update velocities and positions

    for i in range(POP_SIZE):

        for d in range(BIT_LEN):

            r1 = random.random()
            r2 = random.random()

            cognitive = (
                C1
                * r1
                * (
                    pbest_pos[i, d]
                    - particles_pos[i, d]
                )
            )

            social = (
                C2
                * r2
                * (
                    gbest_pos[d]
                    - particles_pos[i, d]
                )
            )

            particles_vel[i, d] = (
                w * particles_vel[i, d]
                + cognitive
                + social
            )

            # Clamp velocity

            if particles_vel[i, d] > V_MAX:

                particles_vel[i, d] = V_MAX

            elif particles_vel[i, d] < V_MIN:

                particles_vel[i, d] = V_MIN

            # Update position

            particles_pos[i, d] += (
                particles_vel[i, d]
            )


# 10. Final Result

print(
    "\n==== Binary PSO Search Completed ===="
)

print(
    "Best fitness (MSE):",
    gbest_fit
)

(
    batch_size,
    opt_name,
    lr,
    n_layers,
    kernels,
    ksize,
    act_name
) = gbest_cfg

print("Best configuration:")

print(
    "  Batch size  :",
    batch_size
)

print(
    "  Optimizer   :",
    opt_name
)

print(
    "  Learning rate:",
    lr
)

print(
    "  Layers      :",
    n_layers
)

print(
    "  Kernels     :",
    kernels
)

print(
    "  Kernel size :",
    ksize
)

print(
    "  Activation  :",
    act_name
)


# Save best configuration

best_config = [
    n_layers,
    ksize,
    kernels,
    act_name,
    lr,
    opt_name,
    batch_size
]

np.save(
    "best_config_evoaae_pso_raw.npy",
    np.array(
        best_config,
        dtype=object
    )
)

print(
    "\nSaved "
    "best_config_evoaae_pso_raw.npy "
    "with:",
    best_config
)
