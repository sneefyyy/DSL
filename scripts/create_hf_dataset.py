#!/usr/bin/env python3
"""Wrapper that runs src.dsl.create_hf_dataset as a script."""
import runpy

if __name__ == '__main__':
    runpy.run_module('src.dsl.create_hf_dataset', run_name='__main__')
