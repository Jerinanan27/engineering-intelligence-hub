"""Engineering Intelligence Hub — a developer-focused hybrid RAG system.

Heavy ML dependencies are imported lazily so that pure-logic modules (and the
CI test job) can be used without installing torch / qdrant.
"""
__all__ = ["EngineeringIntelligenceHub", "Config", "SourceType"]
__version__ = "0.1.0"


def __getattr__(name):  # PEP 562 lazy attribute import
    if name == "EngineeringIntelligenceHub":
        from .pipeline import EngineeringIntelligenceHub
        return EngineeringIntelligenceHub
    if name == "Config":
        from .config import Config
        return Config
    if name == "SourceType":
        from .schema import SourceType
        return SourceType
    raise AttributeError(f"module 'eih' has no attribute {name!r}")
