#!/usr/bin/env python3
# start the app from here
#
# python run.py --devices                  see what mics/speakers you have
# python run.py --setup                    makes the config file for you
# python run.py --selftest samples/x.wav   test it without a microphone
# python run.py --loopback                 hear yourself, checks the wiring
# python run.py                            actually run it

import sys

from interpreter.ui.cli import main

if __name__ == "__main__":
    sys.exit(main())
