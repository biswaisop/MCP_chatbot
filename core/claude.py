"""Backward compatibility module for Claude provider, now replaced with GroqProvider."""
from core.groq import GroqProvider as Claude, GroqProvider, Groq

__all__ = ["Claude", "GroqProvider", "Groq"]
