#!/usr/bin/env python3
"""Public entry point for the reproducible six-component Qwen3-TTS CoreML export."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    source = Path(__file__).resolve().parent / "qwen3_tts_coreml"
    sys.path.insert(0, str(source))
    runpy.run_path(str(source / "convert_coreml.py"), run_name="__main__")
