#!/usr/bin/env python3
"""Wrapper that runs src.dsl.upload_to_hf as a script."""
import runpy

if __name__ == '__main__':
    runpy.run_module('src.dsl.upload_to_hf', run_name='__main__')
