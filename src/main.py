#!/usr/bin/env python

import os
import run

def main():
    os.makedirs("output", exist_ok=True)
    run.run()


if __name__ == "__main__":
    main()
