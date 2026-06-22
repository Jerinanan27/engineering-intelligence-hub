"""Ingest a directory of sources into the hub. Usage: python scripts/ingest.py data/sample"""
import sys
from eih import EngineeringIntelligenceHub

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample"
    hub = EngineeringIntelligenceHub()
    stats = hub.ingest(path)
    print(f"Ingested {stats['documents']} documents -> {stats['chunks']} chunks")
    for t, n in stats.get("by_type", {}).items():
        print(f"  {t:10s}: {n} chunks")

if __name__ == "__main__":
    main()
