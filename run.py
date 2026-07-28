#!/usr/bin/env python3
"""duplex-interpreter entry point.

    python run.py --devices                     list audio devices
    python run.py --setup                       interactive setup wizard
    python run.py --selftest samples/x.wav      full pipeline, no microphone
    python run.py --loopback                    check device wiring
    python run.py                               run the conversation
"""

from __future__ import annotations

import sys

from interpreter.ui.cli import main

if __name__ == "__main__":
    sys.exit(main())
