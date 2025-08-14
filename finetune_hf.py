"""
Compatibility wrapper: run src.dsl.finetune_hf as a module.
"""

if __name__ == "__main__":
    import runpy
    runpy.run_module("src.dsl.finetune_hf", run_name="__main__")
