"""
Answer Engine RAG layer — build, query, and evaluate retrieval.

Usage (PowerShell):
  python rag_pipeline.py ingest --docs C:\\path\\to\\crawled_pages --config A
  python rag_pipeline.py ingest --docs C:\\path\\to\\crawled_pages --config B
  python rag_pipeline.py ask  --config A --question "What services does this business offer?"
  python rag_pipeline.py eval --questions questions.json

Configs (the chunking tradeoff you'll talk about in interviews):
  A = ~500-token chunks (400 words, 40-word overlap)   -> precise, more chunks
  B = ~1000-token chunks (800 words, 80-word overlap)  -> more context per hit, fewer chunks

Dependencies:  pip install chromadb sentence-transformers anthropic
Set ANTHROPIC_API_KEY in your environment for the `ask` command.
"""

import argparse, functools, json, os, re, sys, pathlib

MODEL = "claude-sonnet-5-5"
EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "rag_db"
TOP_K = 5

CONFIGS = {
    "A": {"words": 400, "overlap": 40},   # ~500 tokens
    "B": {"words": 800, "overlap": 80},   # ~1000 tokens
}

# ---------- corpus loading ----------

TAG_RE = re.compile(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>|<[^>]+>")

def html_to_text(raw: str) -> str:
    text = TAG_RE.sub(" ", raw)
    return re.sub(r"\s+", " ", text).strip()

def load_docs(folder: str):
    """Yields (doc_id, text) for every .txt/.md/.html file under folder."""
    base = pathlib.Path(folder)
    exts = {".txt", ".md", ".html", ".htm"}
    files = [p for p in base.rglob("*") if p.suffix.lower() in exts]
    if not files:
        sys.exit(f"No .txt/.md/.html files found under {folder}")
    for p in files:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        text = html_to_text(raw) if p.suffix.lower() in {".html", ".htm"} else raw
        if len(text.split()) > 30:                       # skip empty shells
            yield str(p.relative_to(base)), text

def chunk(text: str, words: int, overlap: int):
    toks = text.split()
    step = max(words - overlap, 1)
    for start in range(0, len(toks), step):
        piece = toks[start:start + words]
        if len(piece) < 25:
            break
        yield " ".join(piece)

# ---------- embedding + store ----------

def get_collection(config: str, create: bool = False):
    import chromadb
    client = chromadb.PersistentClient(path=DB_PATH)
    name = f"answer_engine_{config}"
    if create:
        try:
            client.delete_collection(name)
        except Exception:
            pass
        return client.create_collection(name, metadata={"hnsw:space": "cosine"})
    return client.get_collection(name)

@functools.lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)

def cmd_ingest(args):
    cfg = CONFIGS[args.config]
    model = embedder()
    col = get_collection(args.config, create=True)
    ids, docs, metas = [], [], []
    n_docs = 0
    for doc_id, text in load_docs(args.docs):
        n_docs += 1
        for i, piece in enumerate(chunk(text, cfg["words"], cfg["overlap"])):
            ids.append(f"{doc_id}::{i}")
            docs.append(piece)
            metas.append({"source": doc_id, "chunk": i})
    print(f"Embedding {len(docs)} chunks from {n_docs} documents (config {args.config}: "
          f"{cfg['words']}w/{cfg['overlap']}w overlap)...")
    BATCH = 128
    for s in range(0, len(docs), BATCH):
        embs = model.encode(docs[s:s+BATCH], show_progress_bar=False).tolist()
        col.add(ids=ids[s:s+BATCH], documents=docs[s:s+BATCH],
                embeddings=embs, metadatas=metas[s:s+BATCH])
        print(f"  {min(s+BATCH, len(docs))}/{len(docs)}")
    print(f"Done. Collection answer_engine_{args.config} ready in ./{DB_PATH}")

# ---------- retrieval ----------

def retrieve(question: str, config: str, k: int = TOP_K):
    model = embedder()
    col = get_collection(config)
    q = model.encode([question]).tolist()
    res = col.query(query_embeddings=q, n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "source": meta["source"], "chunk": meta["chunk"],
                     "score": round(1 - dist, 3)})
    return hits

def cmd_ask(args):
    hits = retrieve(args.question, args.config)
    context = "\n\n".join(f"[{i+1}] (source: {h['source']})\n{h['text']}"
                          for i, h in enumerate(hits))
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=MODEL, max_tokens=600,
        system=("Answer ONLY from the numbered context passages. Cite passage numbers "
                "like [1]. If the context does not contain the answer, say exactly: "
                "'Not found in the crawled content.' Do not use outside knowledge."),
        messages=[{"role": "user",
                   "content": f"Context:\n{context}\n\nQuestion: {args.question}"}],
    )
    print("\n--- Retrieved sources ---")
    for i, h in enumerate(hits):
        print(f"[{i+1}] {h['source']} (chunk {h['chunk']}, score {h['score']})")
    print("\n--- Grounded answer ---")
    print(msg.content[0].text)

# ---------- evaluation ----------

def cmd_eval(args):
    questions = json.loads(pathlib.Path(args.questions).read_text(encoding="utf-8"))
    print(f"Evaluating retrieval hit@{TOP_K} on {len(questions)} questions...\n")
    results = {}
    for config in CONFIGS:
        try:
            get_collection(config)
        except Exception:
            print(f"(config {config} not ingested yet — skipping)")
            continue
        hits_count = 0
        misses = []
        for q in questions:
            hits = retrieve(q["question"], config)
            blob = " ".join(h["text"].lower() for h in hits)
            sources = " ".join(h["source"].lower() for h in hits)
            ok = any(phrase.lower() in blob or phrase.lower() in sources
                     for phrase in q["expect_any"])
            hits_count += ok
            if not ok:
                misses.append(q["question"])
        rate = hits_count / len(questions)
        results[config] = rate
        cfg = CONFIGS[config]
        print(f"Config {config} ({cfg['words']}w/{cfg['overlap']}w): "
              f"hit@{TOP_K} = {hits_count}/{len(questions)} ({rate:.0%})")
        for m in misses:
            print(f"    miss: {m}")
        print()
    if len(results) == 2:
        a, b = results.get("A", 0), results.get("B", 0)
        verdict = ("smaller chunks (A) retrieved better" if a > b else
                   "larger chunks (B) retrieved better" if b > a else
                   "both configs tied")
        print(f"Finding: {verdict} on this corpus — that's your chunking-tradeoff answer.")

# ---------- main ----------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Answer Engine RAG layer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest"); p.add_argument("--docs", required=True)
    p.add_argument("--config", choices=list(CONFIGS), default="A"); p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask"); p.add_argument("--question", required=True)
    p.add_argument("--config", choices=list(CONFIGS), default="A"); p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("eval"); p.add_argument("--questions", default="questions.json", help="copy questions.example.json and fill in phrases from your corpus")
    p.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)
