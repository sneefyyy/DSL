"""
Compatibility wrapper: run src.dsl.upload_to_hf as a module.
"""

if __name__ == "__main__":
    import runpy
    runpy.run_module("src.dsl.upload_to_hf", run_name="__main__")
