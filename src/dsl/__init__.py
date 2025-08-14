"""src.dsl package exports."""
# Keep imports lazy to avoid side-effects at package import time in scripts.
from .interpreter import ExampleInterpreter

__all__ = ["ExampleInterpreter"]
