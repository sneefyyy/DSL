#!/usr/bin/env python3
"""Wrapper that runs the package module src.dsl.evaluate_model as a script."""
import runpy

if __name__ == '__main__':
    runpy.run_module('src.dsl.evaluate_model', run_name='__main__')
