#!/usr/bin/env python3
"""
benchmark-pc.py — run the same workloads as the Android app, but on the host
PC's CUDA GPU (RTX 4080S). Emits a JSON line per result, matching the schema
the Android benchmark screen produces, so you can compare side-by-side.

Setup once:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install diffusers transformers accelerate openai-whisper basicsr realesrgan timm

Run:
    python scripts/benchmark-pc.py --iters 8 --warmup 2

Or just one workload:
    python scripts/benchmark-pc.py --only sd15
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import contextmanager
from typing import Callable

DEVICE = "cuda"


def emit_row(model: str, backend: str, samples_us: list[int], error: str | None = None):
    if error or not samples_us:
        row = dict(modelName=model, backend=backend, iterations=0,
                   medianUs=0, p95Us=0, meanUs=0, minUs=0, maxUs=0, error=error)
    else:
        s = sorted(samples_us)
        row = dict(
            modelName=model, backend=backend, iterations=len(s),
            medianUs=s[len(s) // 2],
            p95Us=s[min(len(s) - 1, int(len(s) * 0.95))],
            meanUs=sum(s) // len(s),
            minUs=s[0], maxUs=s[-1],
            error=None,
        )
    print(json.dumps(row, ensure_ascii=False))
    sys.stdout.flush()


@contextmanager
def cuda_timer():
    import torch
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    yield lambda: int((time.perf_counter_ns() - t0) / 1000)
    torch.cuda.synchronize()


def bench(name: str, iters: int, warmup: int, fn: Callable[[], None]):
    samples = []
    for _ in range(warmup):
        fn()
    for _ in range(iters):
        with cuda_timer() as elapsed:
            fn()
        samples.append(elapsed())
    emit_row(name, "RTX_CUDA", samples)


def bench_sd15(iters: int, warmup: int):
    import torch
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None
    prompt = "a moody cyberpunk fox in neon rain, cinematic, sharp focus"
    # Full pipeline (text encoder + 20-step UNet + VAE) — match Android numbers
    fn = lambda: pipe(prompt, num_inference_steps=20, height=512, width=512)
    bench("Stable Diffusion 1.5", iters, warmup, fn)


def bench_whisper(iters: int, warmup: int):
    import torch, whisper, numpy as np
    model = whisper.load_model("base", device=DEVICE)
    audio = np.zeros(16000 * 30, dtype=np.float32)  # 30s silence
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
    fn = lambda: model.transcribe_audio(mel) if hasattr(model, "transcribe_audio") \
        else model.decode(model.encoder(mel.unsqueeze(0)),
                          whisper.DecodingOptions(without_timestamps=True))
    bench("Whisper Base", iters, warmup, fn)


def bench_esrgan(iters: int, warmup: int):
    import torch
    from realesrgan import RealESRGANer
    # RealESRGAN model — see realesrgan README for the exact init.
    # If unavailable, fall back to a placeholder timing of a 256→1024 upscaler.
    sr = RealESRGANer(scale=4, model_path="weights/RealESRGAN_x4plus.pth",
                      half=True, gpu_id=0)
    import numpy as np
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    fn = lambda: sr.enhance(img, outscale=4)
    bench("Real-ESRGAN x4", iters, warmup, fn)


def bench_mobilenet(iters: int, warmup: int):
    import torch, timm
    model = timm.create_model("mobilenetv3_large_100", pretrained=True).to(DEVICE).eval().half()
    x = torch.randn(1, 3, 224, 224, device=DEVICE, dtype=torch.float16)
    fn = lambda: model(x)
    bench("MobileNet-V3 Large", iters, warmup, fn)


BENCHES = {
    "sd15": bench_sd15,
    "whisper": bench_whisper,
    "esrgan": bench_esrgan,
    "mobilenet": bench_mobilenet,
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--only", choices=list(BENCHES), default=None,
                   help="run a single benchmark instead of all of them")
    args = p.parse_args()
    keys = [args.only] if args.only else list(BENCHES)
    for k in keys:
        try:
            BENCHES[k](args.iters, args.warmup)
        except Exception as e:
            emit_row(k, "RTX_CUDA", [], error=str(e))


if __name__ == "__main__":
    main()
