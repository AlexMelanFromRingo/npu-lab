#!/usr/bin/env python3
"""
fetch-models.py — get QNN context binaries for NPU Lab.

Two paths:

  --source=public  (DEFAULT, no account needed)
      Downloads pre-compiled context binaries directly from Qualcomm's public
      HuggingFace S3 bucket. Works for Stable Diffusion 1.5 and Whisper Base.

  --source=aihub
      Submits compile jobs to Qualcomm AI Hub. Required for models that don't
      ship a pre-built context binary on the public bucket (Real-ESRGAN x4,
      MobileNet-V3 — only QNN_DLC is public; we need w8a16 context binary).
      Needs a free account at https://aihub.qualcomm.com.

Setup once:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r scripts/requirements.txt

After it finishes, push everything to the phone:
    adb push models/ /sdcard/Android/data/io.melan.npulab/files/

Usage:
    # Default: SD 1.5 + Whisper from public S3, no account
    python scripts/fetch-models.py

    # Specific list
    python scripts/fetch-models.py --models sd15 whisper

    # AI Hub path for ESRGAN/MobileNet (needs `qai-hub configure`)
    python scripts/fetch-models.py --source aihub --models esrgan mobilenet
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

# Target chip: Snapdragon 8 Elite Gen 5 (Galaxy S26 Ultra).
TARGET_DEVICE = "Samsung Galaxy S26 Ultra"
TARGET_SLUG = "qualcomm_snapdragon_8_elite_gen5"
RELEASE = "v0.54.0"


@dataclass(frozen=True)
class PublicAsset:
    """One zipfile on the public S3 bucket. May contain multiple .bin files."""
    url: str
    # Map of files-inside-zip -> destination-relative-to-models/
    # Use trailing "*" in src to copy any file matching that prefix.
    members: dict[str, str]


PUBLIC_RECIPES: dict[str, PublicAsset] = {
    "sd15": PublicAsset(
        url=f"https://qaihub-public-assets.s3.us-west-2.amazonaws.com/"
            f"qai-hub-models/models/stable_diffusion_v1_5/releases/{RELEASE}/"
            f"stable_diffusion_v1_5-qnn_context_binary-w8a16-{TARGET_SLUG}.zip",
        # The zip from AI Hub usually contains all three stages as .bin files
        # with names like text_encoder.bin / unet.bin / vae_decoder.bin (the
        # exact names may have a prefix; we look for the suffixes).
        members={
            "*text_encoder*.bin": "sd15/text_encoder.bin",
            "*unet*.bin": "sd15/unet.bin",
            "*vae*.bin": "sd15/vae_decoder.bin",
        },
    ),
    "whisper": PublicAsset(
        url=f"https://qaihub-public-assets.s3.us-west-2.amazonaws.com/"
            f"qai-hub-models/models/whisper_base/releases/{RELEASE}/"
            f"whisper_base-qnn_context_binary-float-{TARGET_SLUG}.zip",
        members={
            "*encoder*.bin": "whisper/encoder.bin",
            "*decoder*.bin": "whisper/decoder.bin",
        },
    ),
    "whisper_small": PublicAsset(
        url=f"https://qaihub-public-assets.s3.us-west-2.amazonaws.com/"
            f"qai-hub-models/models/whisper_small/releases/{RELEASE}/"
            f"whisper_small-qnn_context_binary-float-{TARGET_SLUG}.zip",
        members={
            "*encoder*.bin": "whisper_small/encoder.bin",
            "*decoder*.bin": "whisper_small/decoder.bin",
        },
    ),
    "whisper_turbo": PublicAsset(
        url=f"https://qaihub-public-assets.s3.us-west-2.amazonaws.com/"
            f"qai-hub-models/models/whisper_large_v3_turbo/releases/{RELEASE}/"
            f"whisper_large_v3_turbo-qnn_context_binary-float-{TARGET_SLUG}.zip",
        members={
            "*encoder*.bin": "whisper_large_v3_turbo/encoder.bin",
            "*decoder*.bin": "whisper_large_v3_turbo/decoder.bin",
        },
    ),
}

# tokenizer vocab per whisper variant: (models-key, HF repo, dest dir)
WHISPER_TOKENIZERS = {
    "whisper": ("openai/whisper-base", "whisper"),
    "whisper_small": ("openai/whisper-small", "whisper_small"),
    "whisper_turbo": ("openai/whisper-large-v3-turbo", "whisper_large_v3_turbo"),
}


def fetch_url(url: str) -> bytes:
    """Stream-download with progress."""
    print(f"  GET {url}")
    with urllib.request.urlopen(url) as resp:
        total = int(resp.headers.get("Content-Length", "0"))
        chunks = []
        got = 0
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            chunks.append(chunk)
            got += len(chunk)
            if total:
                pct = (got * 100) // total
                print(f"\r  {got // (1 << 20):>4} / {total // (1 << 20):>4} MiB "
                      f"({pct:3d}%)", end="", flush=True)
        print()
        return b"".join(chunks)


def fnmatch_simple(name: str, pattern: str) -> bool:
    """Tiny wildcard match: only * as a glob."""
    if "*" not in pattern:
        return name == pattern
    parts = pattern.split("*")
    pos = 0
    if parts[0] and not name.startswith(parts[0]):
        return False
    pos = len(parts[0])
    for p in parts[1:-1]:
        idx = name.find(p, pos)
        if idx < 0:
            return False
        pos = idx + len(p)
    if parts[-1] and not name.endswith(parts[-1]):
        return False
    return True


def fetch_public(name: str, asset: PublicAsset) -> None:
    print(f"\n=== {name} (public) ===")
    raw = fetch_url(asset.url)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        for pattern, dest in asset.members.items():
            matches = [n for n in names if fnmatch_simple(n.split("/")[-1], pattern)]
            if not matches:
                # Try matching the full path.
                matches = [n for n in names if fnmatch_simple(n, pattern)]
            if not matches:
                print(f"  !! no member matched {pattern!r} (zip has: "
                      f"{', '.join(names[:5])}{'…' if len(names) > 5 else ''})")
                continue
            # Prefer the largest match (usually the actual model, not a sidecar)
            src = max(matches, key=lambda m: zf.getinfo(m).file_size)
            out_path = MODELS_DIR / dest
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and out_path.stat().st_size == zf.getinfo(src).file_size:
                print(f"  [skip] {dest} ({out_path.stat().st_size // (1 << 20)} MiB)")
                continue
            with zf.open(src) as fsrc, open(out_path, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
            print(f"  -> {dest}  ({out_path.stat().st_size // (1 << 20)} MiB)  "
                  f"from {src}")


# ---- HF tokenizer downloads ----

def fetch_clip_tokenizer() -> None:
    out_dir = MODELS_DIR / "sd15" / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("vocab.json", "merges.txt"):
        target = out_dir / fname
        if target.exists():
            print(f"  [skip] tokenizer {fname}")
            continue
        url = f"https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/{fname}"
        print(f"  GET tokenizer {fname}")
        with urllib.request.urlopen(url) as resp:
            target.write_bytes(resp.read())
        print(f"  -> {target}")


def fetch_whisper_tokenizer(key: str) -> None:
    # Decode-only: the app needs just vocab.json (reverse byte-level BPE).
    hf_repo, dest_dir = WHISPER_TOKENIZERS[key]
    out_dir = MODELS_DIR / dest_dir / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "vocab.json"
    if target.exists():
        print(f"  [skip] {dest_dir} tokenizer vocab.json")
        return
    url = f"https://huggingface.co/{hf_repo}/resolve/main/vocab.json"
    print(f"  GET {dest_dir} tokenizer vocab.json")
    with urllib.request.urlopen(url) as resp:
        target.write_bytes(resp.read())
    print(f"  -> {target}")


# ---- AI Hub path (for models without public context_binary) ----

def fetch_aihub(name: str, hub_model_id: str, output_rel: str) -> None:
    """Compile a model via AI Hub for our target device, download the .bin."""
    try:
        import qai_hub as hub
    except ImportError:
        sys.stderr.write("ERROR: qai-hub not installed. pip install qai-hub qai-hub-models\n")
        sys.exit(1)

    print(f"\n=== {name} (AI Hub) ===")
    out_path = MODELS_DIR / output_rel
    if out_path.exists():
        print(f"  [skip] {output_rel} exists")
        return

    model_pkg = __import__(f"qai_hub_models.models.{hub_model_id}", fromlist=["Model"])
    torch_model = model_pkg.Model.from_pretrained()
    sample_inputs = torch_model.sample_inputs()

    print(f"  compiling on '{TARGET_DEVICE}'…")
    job = hub.submit_compile_job(
        model=torch_model.convert_to_torchscript(),
        input_specs=sample_inputs,
        device=hub.Device(TARGET_DEVICE),
        options="--target_runtime qnn_context_binary --quantize_full_type w8a16",
    )
    print(f"    job_id={job.job_id}, watch progress at {job.url}")
    # get_target_model() blocks until the job completes and returns the
    # compiled artifact (job.wait() returns only a status object).
    compiled = job.get_target_model()
    if compiled is None:
        raise RuntimeError(f"compile job failed — see {job.url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.download(str(out_path))
    print(f"  -> {output_rel}  ({out_path.stat().st_size // (1 << 20)} MiB)")


AIHUB_RECIPES = {
    "esrgan":    ("real_esrgan_x4plus",   "real_esrgan_x4.bin"),
    "mobilenet": ("mobilenet_v3_large",   "mobilenet_v3.bin"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=("public", "aihub"), default="public",
                   help="public = no account, just HuggingFace S3 mirror; "
                        "aihub = compile via Qualcomm AI Hub (needs configure)")
    p.add_argument("--models", nargs="+", default=None,
                   help="subset of: sd15, whisper, esrgan, mobilenet")
    args = p.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    if args.source == "public":
        keys = args.models or list(PUBLIC_RECIPES.keys())
        for k in keys:
            if k not in PUBLIC_RECIPES:
                print(f"!! '{k}' has no public mirror — try --source aihub",
                      file=sys.stderr)
                continue
            fetch_public(k, PUBLIC_RECIPES[k])
        if "sd15" in keys:
            fetch_clip_tokenizer()
        for key in keys:
            if key in WHISPER_TOKENIZERS:
                fetch_whisper_tokenizer(key)
    else:
        keys = args.models or list(AIHUB_RECIPES.keys())
        for k in keys:
            if k not in AIHUB_RECIPES:
                print(f"!! '{k}' is fetched from public mirror — re-run "
                      f"without --source aihub", file=sys.stderr)
                continue
            hub_id, rel = AIHUB_RECIPES[k]
            try:
                fetch_aihub(k, hub_id, rel)
            except Exception as e:
                print(f"!! {k}: {e}", file=sys.stderr)

    print(f"\nDone. Push to device with:")
    print(f"  adb push {MODELS_DIR} /sdcard/Android/data/io.melan.npulab/files/")


if __name__ == "__main__":
    main()
