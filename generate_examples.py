"""
Compatibility wrapper: run src.dsl.generate_examples as a module.
"""

if __name__ == "__main__":
    import runpy
    runpy.run_module("src.dsl.generate_examples", run_name="__main__")
from generators.mirror_shape import MirrorShapeGenerator
