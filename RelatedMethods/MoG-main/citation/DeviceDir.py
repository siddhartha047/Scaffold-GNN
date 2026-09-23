"""Portable paths and automatic CUDA selection."""
import os
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIR = os.environ.get("SCAFFOLD_DATA_ROOT", "./data")
