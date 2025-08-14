#!/usr/bin/env python3
"""Wrapper script that calls src.dsl.finetune_hf.main() to keep CLI at repo root."""
from src.dsl import finetune_hf

if __name__ == '__main__':
    finetune_hf.main()
