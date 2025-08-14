# generators/__init__.py
"""
Auto-import all generator classes from the generators package.
"""
import os
import importlib
from pathlib import Path

# Get the directory of this file
package_dir = Path(__file__).parent

# Dictionary to store all generator classes
__all__ = []

# Automatically import all Python files in the package
for file in package_dir.glob("*.py"):
    if file.name.startswith("_") or file.name == "base.py":
        continue
    
    module_name = file.stem
    module = importlib.import_module(f".{module_name}", package=__name__)
    
    # Find all classes ending with 'Generator' in the module
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and attr_name.endswith('Generator'):
            # Make the class available at package level
            globals()[attr_name] = attr
            __all__.append(attr_name)

# Also export the base class
from .base import ExampleGenerator
__all__.append('ExampleGenerator')
