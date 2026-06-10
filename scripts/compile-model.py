#!/usr/bin/env python3
"""
compile-model.py — превращает СВОЮ модель (ONNX / TorchScript) в QNN context
binary под Hexagon NPU Galaxy S26 Ultra (SM8850, HTP v81) и кладёт результат в
models/custom/<имя>.bin. Всё, что лежит в models/custom/, приложение само
показывает на вкладке Benchmark.

Два пути:

  --source aihub   (ДЕФОЛТ, рекомендуется)
      Компиляция в облаке Qualcomm AI Hub: правильные опции квантизации,
      компилятор всегда свежий, бинарник гарантированно совместим с устройством.
      Нужен бесплатный аккаунт: https://aihub.qualcomm.com → Settings → API token,
      затем один раз:  .venv/bin/qai-hub configure --api_token <ТОКЕН>

  --source local
      Полностью оффлайн через локальный QNN SDK:
      qairt-converter (ONNX → DLC) + qnn-context-binary-generator (DLC → .bin).
      Требует Python 3.10 для нативных биндингов SDK — скрипт сам создаст
      .venv-qairt через `uv venv --python 3.10` (uv качает CPython локально,
      системный Python не трогается). FP16-граф; квантизация локально требует
      калибровочных данных (--input_list, см. доки SDK) и здесь не автоматизирована.

Примеры:
    # ONNX → NPU (fp16, размеры зафиксированы в самом ONNX)
    python scripts/compile-model.py my_model.onnx

    # с квантизацией w8a16 на AI Hub
    python scripts/compile-model.py my_model.onnx --quantize w8a16

    # локально, без облака
    python scripts/compile-model.py my_model.onnx --source local

    # потом на телефон:
    adb push models/custom /sdcard/Android/data/io.melan.npulab/files/models/
    #   (или через вкладку Models → пока только adb/файловый менеджер с ПК)

Ограничения HTP, о которых стоит помнить:
  * статические шейпы — динамические оси зафиксируй при экспорте ONNX;
  * не все операторы покрыты HTP (упадёт на этапе компиляции — смотри лог);
  * context binary привязан к SoC и мажорной линии QNN — для другого чипа
    компилируй заново (см. docs/10-custom-models.md).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CUSTOM_DIR = REPO_ROOT / "models" / "custom"

TARGET_DEVICE = "Samsung Galaxy S26 Ultra"
TARGET_CHIPSET_ATTR = "chipset:qualcomm-snapdragon-8-elite-gen5"
SOC_MODEL = 87        # QNN_SOC_MODEL_SM8850
DSP_ARCH = "v81"


# ───────────────────────── AI Hub route ─────────────────────────

def compile_aihub(model_path: Path, out_path: Path, quantize: str | None,
                  device_name: str) -> None:
    try:
        import qai_hub as hub
    except ImportError:
        sys.exit("ERROR: pip install qai-hub  (потом qai-hub configure --api_token …)")

    options = "--target_runtime qnn_context_binary"
    if quantize:
        options += f" --quantize_full_type {quantize} --quantize_io"

    try:
        device = hub.Device(device_name)
    except Exception:
        device = hub.Device(attributes=TARGET_CHIPSET_ATTR)

    print(f"[aihub] compile {model_path.name} for '{device_name}'")
    print(f"[aihub] options: {options}")
    job = hub.submit_compile_job(
        model=str(model_path),
        device=device,
        options=options,
    )
    print(f"[aihub] job {job.job_id} → {job.url}")
    target = job.get_target_model()          # blocks until the job finishes
    if target is None:
        sys.exit(f"FAILED — смотри лог компиляции: {job.url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target.download(str(out_path))
    print(f"[aihub] -> {out_path}  ({out_path.stat().st_size // (1 << 20)} MiB)")


def list_devices() -> None:
    try:
        import qai_hub as hub
    except ImportError:
        sys.exit("ERROR: pip install qai-hub")
    for d in hub.get_devices():
        print(f"  {d.name}   [{', '.join(d.attributes)}]")


# ───────────────────────── local route ─────────────────────────

def qnn_sdk_root() -> Path:
    env = os.environ.get("QNN_SDK_ROOT")
    if env and (Path(env) / "include/QNN").is_dir():
        return Path(env)
    lp = REPO_ROOT / "android-app" / "local.properties"
    if lp.exists():
        for line in lp.read_text().splitlines():
            if line.startswith("qnn.sdk.root="):
                p = Path(line.split("=", 1)[1].strip())
                if (p / "include/QNN").is_dir():
                    return p
    sys.exit("ERROR: QNN_SDK_ROOT не найден (env или local.properties)")


def ensure_py310_venv() -> Path:
    """SDK-биндинги требуют CPython 3.10 — поднимаем отдельный venv через uv."""
    venv = REPO_ROOT / ".venv-qairt"
    py = venv / "bin" / "python"
    if not py.exists():
        uv = shutil.which("uv")
        if not uv:
            sys.exit(
                "ERROR: для --source local нужен Python 3.10.\n"
                "Поставь uv (curl -LsSf https://astral.sh/uv/install.sh | sh) — "
                "он скачает CPython 3.10 локально, не трогая систему."
            )
        print("[local] создаю .venv-qairt (Python 3.10 через uv)…")
        subprocess.run([uv, "venv", "--python", "3.10", str(venv)], check=True)
        subprocess.run(
            [uv, "pip", "install", "--python", str(py),
             "numpy<2", "onnx", "packaging", "pyyaml", "protobuf"],
            check=True,
        )
    return py


def compile_local(model_path: Path, out_path: Path) -> None:
    sdk = qnn_sdk_root()
    py = ensure_py310_venv()
    build = REPO_ROOT / "build" / "qairt" / model_path.stem
    build.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{sdk}/lib/python"
    # SDK-биндинги линкованы с libpython3.10.so — добавляем lib/ из
    # uv-дистрибутива CPython рядом с библиотеками SDK.
    py_lib = Path(os.path.realpath(py)).parent.parent / "lib"
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
        f"{sdk}/lib/x86_64-linux-clang",
        str(py_lib),
        env.get("LD_LIBRARY_PATH", ""),
    ]))
    env["QNN_SDK_ROOT"] = str(sdk)

    dlc = build / f"{model_path.stem}.dlc"
    print(f"[local] qairt-converter → {dlc.name}")
    subprocess.run(
        [str(py), f"{sdk}/bin/x86_64-linux-clang/qairt-converter",
         "--input_network", str(model_path),
         "--output_path", str(dlc)],
        check=True, env=env,
    )

    # HTP backend extensions: целевой SoC/арка для offline-prepare.
    htp_cfg = build / "htp_config.json"
    htp_cfg.write_text(json.dumps({
        "devices": [{"soc_model": SOC_MODEL, "dsp_arch": DSP_ARCH}],
    }, indent=1))
    gen_cfg = build / "generator_config.json"
    gen_cfg.write_text(json.dumps({
        "backend_extensions": {
            "shared_library_path": "libQnnHtpNetRunExtensions.so",
            "config_file_path": str(htp_cfg),
        },
    }, indent=1))

    print("[local] qnn-context-binary-generator (HTP, SM8850/v81)…")
    subprocess.run(
        [f"{sdk}/bin/x86_64-linux-clang/qnn-context-binary-generator",
         "--backend", f"{sdk}/lib/x86_64-linux-clang/libQnnHtp.so",
         "--model", f"{sdk}/lib/x86_64-linux-clang/libQnnModelDlc.so",
         "--dlc_path", str(dlc),
         "--config_file", str(gen_cfg),
         "--binary_file", model_path.stem,
         "--output_dir", str(build)],
        check=True, env=env,
    )
    produced = build / f"{model_path.stem}.bin"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, out_path)
    print(f"[local] -> {out_path}  ({out_path.stat().st_size // (1 << 20)} MiB)")


# ───────────────────────── entry ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", nargs="?", help="путь к .onnx / .pt (TorchScript)")
    p.add_argument("--source", choices=("aihub", "local"), default="aihub")
    p.add_argument("--quantize", choices=("w8a16", "w8a8", "w4a16"), default=None,
                   help="квантизация (только aihub); без флага — fp16")
    p.add_argument("--device", default=TARGET_DEVICE,
                   help=f"AI Hub устройство (default: {TARGET_DEVICE})")
    p.add_argument("--out", default=None,
                   help="куда положить .bin (default: models/custom/<имя>.bin)")
    p.add_argument("--list-devices", action="store_true",
                   help="показать доступные устройства AI Hub и выйти")
    args = p.parse_args()

    if args.list_devices:
        list_devices()
        return
    if not args.model:
        p.error("укажи модель (.onnx / .pt) или --list-devices")

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        sys.exit(f"ERROR: {model_path} не существует")
    out_path = Path(args.out).resolve() if args.out else CUSTOM_DIR / f"{model_path.stem}.bin"

    if args.source == "aihub":
        compile_aihub(model_path, out_path, args.quantize, args.device)
    else:
        if args.quantize:
            sys.exit("--quantize поддержан только с --source aihub "
                     "(локальная квантизация требует калибровки, см. доки SDK)")
        compile_local(model_path, out_path)

    print("\nГотово. Проверка метаданных:")
    print(f"  $QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-utility \\")
    print(f"      --context_binary {out_path} --json_file /tmp/meta.json")
    print("На телефон:")
    print("  adb push models/custom /sdcard/Android/data/io.melan.npulab/files/models/")
    print("Дальше: вкладка Benchmark → новая модель появится в списке.")


if __name__ == "__main__":
    main()
