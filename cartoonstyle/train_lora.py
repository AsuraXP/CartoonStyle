"""SDXL LoRA trainer for style/character consistency — Kaggle T4 x2 friendly.

Multi-GPU launch (2x T4):
    accelerate launch --num_processes 2 --mixed_precision fp16 \
        -m cartoonstyle.train_lora --data-dir /kaggle/input/my-style --trigger "ckstyle"

Data layout: a folder of images with sidecar .txt captions
    img001.png  img001.txt   <- caption (auto-generated with cartoonstyle.caption, then hand-edited)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


def parse_args():
    p = argparse.ArgumentParser(description="CartoonStyle SDXL LoRA trainer")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default="/kaggle/working/cartoonstyle-lora")
    p.add_argument("--base", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trigger", default=None, help="trigger token prepended to every caption")
    p.add_argument("--train-text-encoders", action="store_true")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--resume", default=None, help="checkpoint dir (peft/ subdir) to resume from")
    p.add_argument("--min-snr-gamma", type=float, default=5.0)
    return p.parse_args()


class CaptionImageDataset(Dataset):
    EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def __init__(self, data_dir: str, width: int, height: int, trigger: str | None = None,
                 max_images: int | None = None):
        self.items: list[tuple[str, str]] = []
        for name in sorted(os.listdir(data_dir)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in self.EXT:
                continue
            img_path = os.path.join(data_dir, name)
            cap_path = os.path.join(data_dir, stem + ".txt")
            if os.path.exists(cap_path):
                caption = open(cap_path, encoding="utf-8").read().strip()
            else:
                caption = stem.replace("_", " ").replace("-", " ")
            if trigger:
                caption = f"{trigger}, {caption}"
            self.items.append((img_path, caption))
            if max_images and len(self.items) >= max_images:
                break
        if not self.items:
            raise SystemExit(f"No images found in {data_dir}")
        self.w, self.h = width, height

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, caption = self.items[i]
        img = Image.open(path).convert("RGB")
        img = ImageOps.fit(img, (self.w, self.h), Image.LANCZOS, centering=(0.5, 0.4))
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        pixel = torch.from_numpy(arr).permute(2, 0, 1)
        return {"pixel_values": pixel, "caption": caption}


def encode_prompts(captions, tokenizer, tokenizer_2, text_encoder, text_encoder_2, device):
    t1 = tokenizer(captions, padding="max_length", max_length=tokenizer.model_max_length,
                   truncation=True, return_tensors="pt").input_ids.to(device)
    t2 = tokenizer_2(captions, padding="max_length", max_length=tokenizer_2.model_max_length,
                     truncation=True, return_tensors="pt").input_ids.to(device)
    out1 = text_encoder(t1, output_hidden_states=True)
    out2 = text_encoder_2(t2, output_hidden_states=True)
    prompt_embeds = torch.cat([out1.hidden_states[-2], out2.hidden_states[-2]], dim=-1)
    pooled = out2[0]
    return prompt_embeds, pooled


def get_lora_state_dict(unet) -> dict:
    from peft.utils import get_peft_model_state_dict

    sd = get_peft_model_state_dict(unet)
    sd = {k.replace("base_model.model.", ""): v for k, v in sd.items()}
    try:
        from diffusers.utils import convert_state_dict_to_diffusers

        sd = convert_state_dict_to_diffusers(sd)
    except Exception:  # noqa: BLE001 - key format is version dependent; keep best effort
        pass
    return sd


def save_lora(unet, text_encoder, text_encoder_2, out_dir: str, args) -> None:
    from diffusers.loaders.lora import StableDiffusionXLLoraLoaderMixin

    os.makedirs(out_dir, exist_ok=True)
    unet_sd = get_lora_state_dict(unet)
    kwargs = {"unet_lora_layers": unet_sd}
    if args.train_text_encoders:
        try:
            kwargs["text_encoder_lora_layers"] = get_lora_state_dict(text_encoder)
            kwargs["text_encoder_2_lora_layers"] = get_lora_state_dict(text_encoder_2)
        except Exception:  # noqa: BLE001
            pass
    StableDiffusionXLLoraLoaderMixin.save_lora_weights(save_directory=out_dir, **kwargs)
    # also persist the raw peft adapter so --resume is exact
    try:
        unet.save_pretrained(os.path.join(out_dir, "peft"))
    except Exception:  # noqa: BLE001
        pass


def main():
    args = parse_args()
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="fp16")
    set_seed(args.seed)
    if accelerator.is_main_process:
        try:
            with open("/proc/meminfo") as f:
                avail_gb = int([l for l in f if l.startswith("MemAvailable")][0].split()[1]) / 1e6
            print(f"[train] host RAM available: {avail_gb:.1f} GB", flush=True)
        except Exception:
            pass
    weight_dtype = torch.float16
    alpha = args.alpha or args.rank

    if accelerator.is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "training_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"[train] data={args.data_dir} out={args.out_dir} steps={args.steps} rank={args.rank}", flush=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.base, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.base, subfolder="text_encoder", torch_dtype=weight_dtype,
                                                 low_cpu_mem_usage=True)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.base, subfolder="text_encoder_2", torch_dtype=weight_dtype, low_cpu_mem_usage=True)
    # fp32 on purpose: the SDXL VAE overflows in fp16 and poisons the latents with inf/NaN
    vae = AutoencoderKL.from_pretrained(args.base, subfolder="vae", torch_dtype=torch.float32,
                                        low_cpu_mem_usage=True)
    unet = UNet2DConditionModel.from_pretrained(args.base, subfolder="unet", torch_dtype=weight_dtype,
                                                low_cpu_mem_usage=True)
    noise_scheduler = DDPMScheduler.from_pretrained(args.base, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_val", "to_out.0"],
    )
    unet = get_peft_model(unet, lora_config)
    unet.enable_gradient_checkpointing()
    try:
        unet.enable_xformers_memory_efficient_attention()  # no-op / safe to skip on T4 stacks
    except Exception:  # noqa: BLE001
        pass

    if args.resume:
        from peft import set_peft_model_state_dict
        from peft.utils import load_peft_weights

        peft_dir = args.resume if os.path.isdir(args.resume) else os.path.join(args.resume, "peft")
        set_peft_model_state_dict(unet, load_peft_weights(peft_dir))
        if accelerator.is_main_process:
            print(f"[train] resumed adapter weights from {peft_dir}", flush=True)

    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.to(torch.float32)  # fp32 master weights (autocast handles the rest)
    if args.train_text_encoders:
        te_config = LoraConfig(r=args.rank, lora_alpha=alpha, init_lora_weights="gaussian",
                               target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
        text_encoder = get_peft_model(text_encoder, te_config)
        text_encoder_2 = get_peft_model(text_encoder_2, te_config)
        params += [p for p in text_encoder.parameters() if p.requires_grad]
        params += [p for p in text_encoder_2.parameters() if p.requires_grad]

    try:
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(params, lr=args.lr)
        if accelerator.is_main_process:
            print("[train] optimizer: AdamW8bit", flush=True)
    except Exception:  # noqa: BLE001
        optimizer = torch.optim.AdamW(params, lr=args.lr)

    dataset = CaptionImageDataset(args.data_dir, args.width, args.height, args.trigger, args.max_images)
    if accelerator.is_main_process:
        print(f"[train] dataset: {len(dataset)} images", flush=True)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False,
                            drop_last=True, collate_fn=lambda b: {
                                "pixel_values": torch.stack([x["pixel_values"] for x in b]),
                                "captions": [x["caption"] for x in b],
                            })
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, dataloader, lr_scheduler)
    if args.train_text_encoders:
        text_encoder, text_encoder_2 = accelerator.prepare(text_encoder, text_encoder_2)

    vae.to(accelerator.device)
    if not args.train_text_encoders:
        text_encoder.to(accelerator.device)
        text_encoder_2.to(accelerator.device)

    gc.collect()
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)
    add_time_ids = torch.tensor(
        [[args.height, args.width, 0, 0, args.height, args.width]], dtype=weight_dtype
    ).to(accelerator.device)

    unet.train()
    step, t0 = 0, time.time()
    data_iter = iter(dataloader)
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float32)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            latents = latents.to(weight_dtype)
        if step == 0 and not torch.isfinite(latents).all():
            raise RuntimeError("VAE produced non-finite latents — check the image folder for corrupt files.")
        if args.train_text_encoders:
            prompt_embeds, pooled = encode_prompts(
                batch["captions"], tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                accelerator.device)
        else:
            with torch.no_grad():
                prompt_embeds, pooled = encode_prompts(
                    batch["captions"], tokenizer, tokenizer_2, text_encoder, text_encoder_2,
                    accelerator.device)
            prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
            pooled = pooled.to(dtype=weight_dtype)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        with accelerator.accumulate(unet):
            model_pred = unet(
                noisy_latents, timesteps, prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids.repeat(bsz, 1)},
            )[0]
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"unsupported prediction_type {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean(dim=[1, 2, 3])
            if args.min_snr_gamma > 0:
                snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
                w = torch.stack([snr, args.min_snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    w = w / snr
                loss = (loss * w).mean()
            else:
                loss = loss.mean()

            if torch.isfinite(loss):
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, 1.0)
                optimizer.step()
                lr_scheduler.step()
            else:
                print(f"[train] WARNING: non-finite loss at step {step}, skipping batch", flush=True)
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            step += 1
            if step % 10 == 0 and accelerator.is_main_process:
                rate = step / max(time.time() - t0, 1e-6)
                eta = (args.steps - step) / max(rate, 1e-6) / 60
                print(f"[train] step {step}/{args.steps} | loss {loss.item():.4f} | "
                      f"{rate:.2f} it/s | eta {eta:.0f}m", flush=True)
            if step % args.save_every == 0 or step == args.steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(unet)
                    save_lora(unwrapped, text_encoder, text_encoder_2,
                              os.path.join(args.out_dir, "checkpoints", f"step_{step:06d}"), args)
                    print(f"[train] saved checkpoint at step {step}", flush=True)
                accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(unet)
        save_lora(unwrapped, text_encoder, text_encoder_2, os.path.join(args.out_dir, "final"), args)
        print(f"[train] done in {(time.time() - t0) / 60:.1f}m -> {os.path.join(args.out_dir, 'final')}", flush=True)


if __name__ == "__main__":
    main()
