#!/usr/bin/env python3
"""Wrapper that runs src.dsl.generate_examples as a script."""
import runpy

if __name__ == '__main__':
    runpy.run_module('src.dsl.generate_examples', run_name='__main__')
