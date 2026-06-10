#!/usr/bin/env python3
"""
inspect-model.py — distill the tensor schema of a QNN context binary on the PC,
WITHOUT touching the device.

This is the verification step you should run BEFORE trying a fresh .bin in
NpuLab. It calls Qualcomm's `qnn-context-binary-utility` from the QNN SDK
(which serialises the binary's metadata to JSON), then prints a compact summary
of every graph's input/output tensors — names, dtypes, shapes.

Useful for:
  - Confirming that StableDiffusionPipeline.runUnet maps inputs correctly
    (UNet expects sample / timestep / encoder_hidden_states in that order).
  - Spotting unexpected dtype mismatches (UInt16 instead of FP16 etc.) that
    would otherwise blow up only on the phone.

Usage:
    export QNN_SDK_ROOT=$HOME/qnn-sdk/qairt/2.46.0.260424
    scripts/inspect-model.py models/sd15/unet.bin
    scripts/inspect-model.py models/sd15/*.bin       # batch
    scripts/inspect-model.py --raw-json models/sd15/unet.bin   # dump full JSON

Requires:
    libc++ on Linux:    sudo apt install -y libc++1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Mirror of Qnn_DataType_t enum from QnnTypes.h. The utility prints these as
# integers in the JSON; we translate so the output is human-readable.
DATATYPE = {
    0x0008: "int8",   0x0016: "int16", 0x0032: "int32", 0x0064: "int64",
    0x0108: "uint8", 0x0116: "uint16", 0x0132: "uint32", 0x0164: "uint64",
    0x0208: "fp8",   0x0216: "fp16",   0x0226: "bf16",   0x0232: "fp32", 0x0264: "fp64",
    0x0308: "sfixed8", 0x0316: "sfixed16", 0x0332: "sfixed32",
    0x0408: "ufixed8", 0x0416: "ufixed16", 0x0432: "ufixed32",
}

def find_utility() -> Path:
    sdk = os.environ.get("QNN_SDK_ROOT")
    if not sdk:
        sys.exit("ERROR: set QNN_SDK_ROOT to your qairt/<version>/ directory")
    candidate = Path(sdk) / "bin" / "x86_64-linux-clang" / "qnn-context-binary-utility"
    if not candidate.exists():
        sys.exit(f"ERROR: tool not found at {candidate}")
    return candidate

def run_utility(tool: Path, bin_path: Path) -> dict:
    sdk = Path(os.environ["QNN_SDK_ROOT"])
    env = os.environ.copy()
    lib_dir = str(sdk / "lib" / "x86_64-linux-clang")
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"

    with tempfile.TemporaryDirectory() as td:
        out_json = Path(td) / "info.json"
        rc = subprocess.run(
            [str(tool),
             f"--context_binary={bin_path}",
             f"--json_file={out_json}",
             "--unified_qairt_format"],
            env=env, capture_output=True, text=True,
        )
        if rc.returncode != 0:
            sys.stderr.write(rc.stdout + "\n" + rc.stderr + "\n")
            sys.exit(f"!! qnn-context-binary-utility failed on {bin_path}")
        return json.loads(out_json.read_text())

def fmt_tensor(t: dict) -> str:
    name = t.get("name", "?")
    dt_raw = t.get("dataType") or t.get("data_type")
    dt = DATATYPE.get(dt_raw, f"dtype={dt_raw}")
    shape = t.get("dimensions") or t.get("shape") or []
    tt = t.get("type", "")
    tt_short = {0: "STATIC", 1: "APP_WR", 2: "APP_RD", 3: "APP_RW", 5: "NATIVE",
                7: "NULL"}.get(tt, str(tt))
    return f"  {name:<48} {dt:>8}  {shape!s:<24}  type={tt_short}"

def walk_graphs(info: dict):
    """qnn-context-binary-utility's output schema varies slightly across QNN versions.
    Look in a few common locations for the graph list."""
    candidates = [
        info.get("graphs"),
        info.get("info", {}).get("graphs"),
        info.get("context", {}).get("graphs"),
        info.get("contextBinaryInfoV3", {}).get("graphs"),
        info.get("contextBinaryInfoV2", {}).get("graphs"),
        info.get("contextBinaryInfoV1", {}).get("graphs"),
    ]
    for g in candidates:
        if g: return g
    return []

def graph_tensors(graph: dict, key_pair: tuple[str, str]) -> list[dict]:
    a, b = key_pair
    # Modern utility: graphInfoV2.graphInputs / .graphOutputs at the top of graph
    for outer in ("graphInfoV3", "graphInfoV2", "graphInfoV1", None):
        src = graph.get(outer) if outer else graph
        if not isinstance(src, dict): continue
        v = src.get(a) or src.get(b)
        if v: return v
    return []

def print_summary(bin_path: Path, info: dict):
    graphs = walk_graphs(info)
    print(f"\n=== {bin_path} ===")
    if not graphs:
        print("  (no graphs found — try --raw-json to inspect output)")
        return
    for i, g in enumerate(graphs):
        gname = (g.get("graphInfoV3") or g.get("graphInfoV2") or
                 g.get("graphInfoV1") or g).get("graphName", f"graph_{i}")
        print(f"  graph[{i}] name='{gname}'")
        ins = graph_tensors(g, ("graphInputs", "inputs"))
        outs = graph_tensors(g, ("graphOutputs", "outputs"))
        print(f"  inputs  ({len(ins)}):")
        for t in ins: print(fmt_tensor(t))
        print(f"  outputs ({len(outs)}):")
        for t in outs: print(fmt_tensor(t))

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bin", nargs="+", help="path(s) to .bin context binary")
    p.add_argument("--raw-json", action="store_true",
                   help="print the full JSON dump for one file (debug)")
    args = p.parse_args()

    tool = find_utility()
    for bp in args.bin:
        path = Path(bp)
        if not path.exists():
            print(f"!! {bp}: not found", file=sys.stderr)
            continue
        info = run_utility(tool, path)
        if args.raw_json:
            print(json.dumps(info, indent=2))
        else:
            print_summary(path, info)

if __name__ == "__main__":
    main()
