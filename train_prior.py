import math
from os import environ, path

import GPUtil as gpu
import numpy as np
import pytorch_lightning as pl
import torch
from effortless_config import Config
from torch.utils.data import DataLoader, random_split
from udls import SimpleDataset, simple_audio_preprocess

from prior.model import Model


class args(Config):
    RESOLUTION = 32

    RES_SIZE = 512
    SKP_SIZE = 256
    KERNEL_SIZE = 3
    CYCLE_SIZE = 4
    N_LAYERS = 10
    PRETRAINED_VAE = None

    PREPROCESSED = None
    WAV = None
    SR = 48000
    N_SIGNAL = 65536

    BATCH = 8
    CKPT = None

    NAME = None


args.parse_args()
assert args.NAME is not None


def get_n_signal(a, m):
    k = a.KERNEL_SIZE
    cs = a.CYCLE_SIZE
    l = a.N_LAYERS

    rf = (k - 1) * sum(2 ** (np.arange(l) % cs)) + 1
    ratio = m.encode_params[-1].item()

    return 2 ** math.ceil(math.log2(rf * ratio))


model = Model(
    resolution=args.RESOLUTION,
    res_size=args.RES_SIZE,
    skp_size=args.SKP_SIZE,
    kernel_size=args.KERNEL_SIZE,
    cycle_size=args.CYCLE_SIZE,
    n_layers=args.N_LAYERS,
    pretrained_vae=args.PRETRAINED_VAE,
)

args.N_SIGNAL = max(args.N_SIGNAL, get_n_signal(args, model.synth))

dataset = SimpleDataset(
    args.PREPROCESSED,
    args.WAV,
    preprocess_function=simple_audio_preprocess(args.SR, args.N_SIGNAL),
    split_set="full",
    transforms=lambda x: x.reshape(1, -1),
)

val = (2 * len(dataset)) // 100
train = len(dataset) - val
train, val = random_split(dataset, [train, val])

train = DataLoader(train, args.BATCH, True, drop_last=True, num_workers=8)
val = DataLoader(val, args.BATCH, False, num_workers=8)

# CHECKPOINT CALLBACKS
validation_checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="validation",
    filename="best",
)
last_checkpoint = pl.callbacks.ModelCheckpoint(filename="last")

CUDA = gpu.getAvailable(maxMemory=0.05)
if len(CUDA):
    environ["CUDA_VISIBLE_DEVICES"] = str(CUDA[0])
    use_gpu = 1
elif torch.cuda.is_available():
    print("Cuda is available but no fully free GPU found.")
    print("Training may be slower due to concurrent processes.")
    use_gpu = 1
else:
    print("No GPU found.")
    use_gpu = 0

trainer = pl.Trainer(
    logger=pl.loggers.TensorBoardLogger(path.join("runs", args.NAME), name="prior"),
    gpus=use_gpu,
    val_check_interval=min(10000, len(train)),
    callbacks=[validation_checkpoint, last_checkpoint],
    resume_from_checkpoint=args.CKPT,
    max_epochs=100000,
)
trainer.fit(model, train, val)
