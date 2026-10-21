"""CartoonStyle: Kaggle T4 x2 batch image generation + LoRA training workflow."""

__version__ = "0.1.0"

# Quiet console: the library's nested/duplicated tqdm bars (Loading weights,
# Downloading bytes, Reconstructing, ...) are noise. The pipeline prints its
# own simple one-line status updates instead.
import logging
import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DIFFUSERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
