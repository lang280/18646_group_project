#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys

# default values
DEFAULT_MAX_TRAIN = "100"
DEFAULT_MAX_TEST  = "10"
DEFAULT_ARCH      = "sm_75"

# source files and outputs
CUDA_SRC   = "new.c"
ORI_SRC    = "ori.c"
CUDA_OUT   = "new.x"
ORI_OUT    = "ori.x"

def prompt_with_default(prompt, default):
    """Prompt the user, return default on empty input."""
    val = input(f"{prompt} [default: {default}]: ").strip()
    return val if val else default

def replace_macro(filename, macro, new_value):
    """
    In-place replace a line starting with
      #define MACRO ...
    to
      #define MACRO new_value
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: '{filename}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(filename, 'w') as f:
        for line in lines:
            if line.startswith(f"#define {macro} "):
                f.write(f"#define {macro} {new_value}\n")
            else:
                f.write(line)

def compile_cuda(arch):
    """Compile CUDA sources with nvcc."""
    cmd = [
        "nvcc",
        "-o", CUDA_OUT,
        CUDA_SRC, "forward.cu", "backward.cu",
        "-lm",
        f"-arch={arch}"
    ]
    print("Compiling CUDA:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("nvcc compilation failed.", file=sys.stderr)
        sys.exit(res.returncode)
    print(f"Generated {CUDA_OUT}")

def compile_ori():
    """Compile serial source with gcc."""
    cmd = [
        "gcc",
        "-o", ORI_OUT,
        ORI_SRC,
        "-lm"
    ]
    print("Compiling serial:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("gcc compilation failed.", file=sys.stderr)
        sys.exit(res.returncode)
    print(f"Generated {ORI_OUT}")

def main():
    # 1. prompt for values
    max_train = prompt_with_default("Enter MAX_TRAIN", DEFAULT_MAX_TRAIN)
    max_test  = prompt_with_default("Enter MAX_TEST",  DEFAULT_MAX_TEST)
    arch      = prompt_with_default("Enter GPU -arch", DEFAULT_ARCH)

    # 2. patch both sources
    for src in (CUDA_SRC, ORI_SRC):
        replace_macro(src, "MAX_TRAIN", max_train)
        replace_macro(src, "MAX_TEST",  max_test)
    print(f"Patched MAX_TRAIN={max_train}, MAX_TEST={max_test} in {CUDA_SRC} and {ORI_SRC}")

    # 3. compile
    compile_cuda(arch)
    compile_ori()

if __name__ == "__main__":
    main()
