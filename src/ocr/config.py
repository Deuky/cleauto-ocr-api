import pickle
import torch
import os

VERSION = "v1"
DIR = f"models/{VERSION}"

with open(f"{DIR}/char_to_idx.pkl", "rb") as f:
    CHAR_TO_IDX = pickle.load(f)

with open(f"{DIR}/idx_to_char.pkl", "rb") as f:
    IDX_TO_CHAR = pickle.load(f)

IMAGE_HEIGHT = 32
N_CHANNELS = 1
N_CLASSES = len(CHAR_TO_IDX) + 1
MODEL_FILE = f"{DIR}/model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
