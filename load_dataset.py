"""
Compatibility wrapper: run src.dsl.load_dataset as a module.
"""

if __name__ == "__main__":
    import runpy
    runpy.run_module("src.dsl.load_dataset", run_name="__main__")
    