#!/usr/bin/env python3
"""cutest-picture-01 entry: copy the finished picture to the output directory.

The picture is entry.png beside this file. This program copies those exact bytes,
unchanged, to $MAKEFUN_OUTPUT_DIR/entry.png (or /output/entry.png when the
variable is unset). Standard library only, deterministic, nothing else is read.

    mkdir -p out && MAKEFUN_OUTPUT_DIR=out python3 main.py
"""
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "entry.png"


def output_dir():
    return Path(os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output")


def main():
    target = output_dir()
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, target / "entry.png")


if __name__ == "__main__":
    main()
