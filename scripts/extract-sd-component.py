#!/usr/bin/env python3
"""
extract-sd-component.py — pull one component out of an A1111 / LDM Stable
Diffusion checkpoint (.safetensors) and export it to ONNX, so it can be fed to
scripts/compile-model.py and run on the NPU.

Proven on real checkpoints (e.g. abyssorangemix3): the text encoder compiles to
a valid SM8850 / V81 context binary end-to-end.

Components:
  text_encoder   CLIP text encoder. Small (~123M), low memory, safe. Verified.
  vae_decoder    VAE decoder (latent → image). Heavy activations at full
                 512x512 — use --vae-size to keep memory bounded.

Memory note: exporting the VAE decoder at 64x64 latent → 512x512 image traces
512x512x512 activations and can need >13 GB RAM (it OOM'd a 15 GB box). Default
--vae-size 256 keeps it safe; bump to 512 only with plenty of RAM.

Usage:
  python scripts/extract-sd-component.py CKPT.safetensors text_encoder
  python scripts/extract-sd-component.py CKPT.safetensors vae_decoder --vae-size 256
  # then:
  python scripts/compile-model.py /tmp/<name>.onnx --source local
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path


def export_text_encoder(src: str, out: str) -> None:
    import torch
    from safetensors import safe_open
    from transformers import CLIPTextModel, CLIPTextConfig

    torch.set_num_threads(2)
    # transformers >=5 flattens CLIPTextModel: state_dict keys drop "text_model.".
    prefix = "cond_stage_model.transformer.text_model."
    sd = {}
    with safe_open(src, framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix) and not k.endswith("position_ids"):
                sd[k[len(prefix):]] = f.get_tensor(k).float()
    if not sd:
        sys.exit("No cond_stage_model.* keys — is this an SD 1.x checkpoint "
                 "(not SDXL/FLUX, not a LoRA)?")
    cfg = CLIPTextConfig(vocab_size=49408, hidden_size=768, intermediate_size=3072,
                         num_hidden_layers=12, num_attention_heads=12,
                         max_position_embeddings=77)
    m = CLIPTextModel(cfg).eval()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        sys.exit(f"weights did not fully load (missing {len(missing)})")
    print(f"[te] loaded {len(sd)} tensors, {sum(p.numel() for p in m.parameters())//10**6}M params")

    class Enc(torch.nn.Module):
        def __init__(self, x): super().__init__(); self.x = x
        def forward(self, ids): return self.x(input_ids=ids).last_hidden_state

    ids = torch.zeros(1, 77, dtype=torch.int64)
    torch.onnx.export(Enc(m).eval(), ids, out, input_names=["input_ids"],
                      output_names=["text_embedding"], opset_version=17)


def export_vae_decoder(src: str, out: str, latent: int) -> None:
    import torch
    from safetensors import safe_open
    from diffusers import AutoencoderKL
    from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
        convert_ldm_vae_checkpoint,
    )

    torch.set_num_threads(2)
    full = {}
    with safe_open(src, framework="pt") as f:
        for k in f.keys():
            if k.startswith("first_stage_model."):
                full[k] = f.get_tensor(k).float()
    if not full:
        sys.exit("No first_stage_model.* (VAE) keys in this checkpoint")
    config = {"sample_size": 512, "in_channels": 3, "out_channels": 3,
              "down_block_types": ["DownEncoderBlock2D"] * 4,
              "up_block_types": ["UpDecoderBlock2D"] * 4,
              "block_out_channels": [128, 256, 512, 512],
              "latent_channels": 4, "layers_per_block": 2}
    vae = AutoencoderKL(**config)
    vae.load_state_dict(convert_ldm_vae_checkpoint(full, config), strict=False)
    vae = vae.eval().float()
    del full; gc.collect()
    print(f"[vae] decoder ready; exporting at {latent}x{latent} latent → "
          f"{latent*8}x{latent*8} image")

    class Dec(torch.nn.Module):
        def __init__(self, v): super().__init__(); self.v = v
        def forward(self, z):
            z = z / self.v.config.scaling_factor
            z = self.v.post_quant_conv(z)
            return (self.v.decoder(z) / 2 + 0.5).clamp(0, 1)

    z = torch.randn(1, 4, latent, latent)
    torch.onnx.export(Dec(vae).eval(), z, out, input_names=["latent"],
                      output_names=["image"], opset_version=17,
                      do_constant_folding=False)  # folding balloons memory


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="SD 1.x .safetensors")
    p.add_argument("component", choices=("text_encoder", "vae_decoder"))
    p.add_argument("--out", default=None, help="output .onnx (default /tmp/<name>.onnx)")
    p.add_argument("--vae-size", type=int, default=256,
                   help="VAE latent H/W (64=512px image, heavy; default 256px)")
    args = p.parse_args()

    src = str(Path(args.checkpoint).resolve())
    if not os.path.exists(src):
        sys.exit(f"{src} not found")
    stem = Path(src).stem
    out = args.out or f"/tmp/{stem}_{args.component}.onnx"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    if args.component == "text_encoder":
        export_text_encoder(src, out)
    else:
        export_vae_decoder(src, out, args.vae_size)

    print(f"\nONNX → {out}  ({os.path.getsize(out)//2**20} MiB)")
    print("Compile for the NPU:")
    print(f"  python scripts/compile-model.py {out} --source local")


if __name__ == "__main__":
    main()
