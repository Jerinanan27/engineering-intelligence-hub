"""Ask the hub a question. Usage: python scripts/query.py "how is auth handled?" """
import sys
from eih import EngineeringIntelligenceHub

def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/query.py "your question"'); return
    question = sys.argv[1]
    hub = EngineeringIntelligenceHub()
    hub.ingest("data/sample")          # in-memory store: ingest then ask
    ans = hub.ask(question)
    print("\n=== ANSWER ===\n" + ans.text)
    print("\n=== SOURCES ===")
    for i, c in enumerate(ans.citations, 1):
        src = c.chunk.metadata.get("path", c.chunk.doc_id)
        print(f"[{i}] {src}  (type={c.chunk.source_type.value}, score={c.score:.3f})")

if __name__ == "__main__":
    main()
