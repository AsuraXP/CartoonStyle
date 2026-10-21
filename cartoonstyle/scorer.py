"""Image/word match scoring (cosine similarity) - no prompts, no yes/no roulette.

Images and texts are encoded SEPARATELY, then compared - there is no pairing
ambiguity at all. Cosine similarity is text-sensitive and stable: same picture
+ words always give the same score. Also returns which text each picture
matches BEST out of the whole batch - the alignment self-check.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

SCORER = "google/siglip-base-patch16-224"


def load_scorer(device: str = ""):
    from transformers import AutoModel, AutoProcessor

    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[load] match scorer {SCORER}", flush=True)
    processor = AutoProcessor.from_pretrained(SCORER)
    model = AutoModel.from_pretrained(SCORER).to(device).eval()
    print("[load] match scorer ready", flush=True)
    return model, processor


@torch.no_grad()
def score_pairs(model, processor, image_paths: list, texts: list, batch: int = 16):
    """Return (paired_scores, best_text_index, own_rank) as plain lists.

    paired_scores[i]   = cosine(image_i, text_i)          the match score
    best_text_index[i] = argmax_j cosine(image_i, text_j) the alignment self-check
    own_rank[i]      = rank of text_i among all texts for image_i (1 = best)
    """
    from PIL import Image

    dev = next(model.parameters()).device
    img_embs, txt_embs = [], []
    for i in range(0, len(image_paths), batch):
        ims = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch]]
        inputs = processor(images=ims, return_tensors="pt").to(dev)
        img_embs.append(F.normalize(model.get_image_features(**inputs), dim=-1))
    for i in range(0, len(texts), batch):
        tx = [" ".join(t.split())[:400] for t in texts[i:i + batch]]
        inputs = processor(text=tx, padding="max_length", return_tensors="pt").to(dev)
        txt_embs.append(F.normalize(model.get_text_features(**inputs), dim=-1))
    imgs = torch.cat(img_embs)
    txts = torch.cat(txt_embs)
    sim = imgs @ txts.T                       # [N, N] cosine, all pairs
    diag = sim.diagonal()
    paired = diag.float().cpu().tolist()
    best = sim.argmax(dim=1).tolist()
    ranks = ((sim > diag.unsqueeze(1)).sum(dim=1) + 1).tolist()   # 1 = own words are the best match
    return paired, best, ranks
