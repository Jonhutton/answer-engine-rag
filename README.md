# Answer Engine — RAG Layer

Retrieval-augmented grounding for [Foundation AI](https://foundation-ai.net)'s
AI-visibility audit product. Embeds a crawled website corpus into a local vector
store, retrieves the most relevant chunks for a question, and has Claude answer
**only from retrieved text, with citations** — refusing when the corpus doesn't
contain the answer.

Built to answer a practical question: when the Answer Engine audits a site,
can retrieval ground its findings in the site's own words instead of whole-page
prompting?

## Pipeline

```
crawled pages (.html/.txt/.md)
      │  chunk: config A (~500 tok, 10% overlap)  |  config B (~1000 tok, 10% overlap)
      ▼
sentence-transformers (all-MiniLM-L6-v2) ──► ChromaDB (cosine, persistent)
      │  top-5 retrieval
      ▼
Claude — answers only from retrieved passages, cites [n], refuses on no-evidence
```

## Run it

```powershell
pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "sk-..."

# 0. crawl a site via its sitemap (honors robots.txt, strips nav/footer + cross-page boilerplate)
python crawl.py --site https://www.example-business.com --max 80

# 1. ingest the corpus twice (two chunking configs)
python rag_pipeline.py ingest --docs .\crawled_pages --config A
python rag_pipeline.py ingest --docs .\crawled_pages --config B

# 2. ask a grounded question
python rag_pipeline.py ask --config A --question "How much is the basic membership if I pay annually?"

# 3. evaluate retrieval (edit questions.json first — 10 questions with known answers)
python rag_pipeline.py eval   # copy questions.example.json -> questions.json and fill in your own
```

## Retrieval evaluation

`eval` measures **hit@5**: for 10 questions with known-correct phrases, how often
does a chunk containing the answer land in the top-5 retrieved? Both chunking
configs are scored side by side.

**Corpus:** 79 pages crawled from a client retailer's public website (a real Answer
Engine audit target, kept anonymous), 74 kept after cleaning. Questions cover facts spread
across the about, contact, location, membership, coupon and FAQ pages; every expected
phrase was checked against the corpus before running eval. The real questions file and
corpus stay local because they identify the client. `questions.example.json` shows the format.

| Config | Chunking | Chunks | hit@5 | Missed |
|--------|----------|--------|-------|--------|
| A | ~500 tokens, 10% overlap | 321 | **9/10** | store hours |
| B | ~1000 tokens, 10% overlap | 178 | 8/10 | loyalty program, meaning of the name |

**Finding:** smaller chunks won 9–8, but the two configs failed on *different*
questions. B lost short, specific facts (a one-line loyalty rule, a two-sentence
name origin) that got diluted inside 800-word chunks of surrounding text. A lost
store hours: those sit in a short address block on a page dominated by
product-category copy, and the semantic embedding ranked promo-heavy pages higher.
That's the precision-vs-context tradeoff showing up in real data, and it points to
the next step: hybrid retrieval (BM25 keyword + embeddings) for exact-fact lookups
like hours, phone numbers and prices.

### Grounded answers (config A, client names redacted)

```
Q: How much is the [basic tier] membership if I pay annually?
A: The [basic tier] membership costs $199.99 if you pay annually. That works out
   to about $16.67 per month and saves 17% [1].            (source: memberships.txt, score 0.71)

Q: Who is the CEO, and what is his shoe size?
A: The CEO is [name], who is also Co-Owner [4]. The context does not give
   his shoe size, so I can't answer that part.

Q: What are the retail store hours?
A: Not found in the crawled content.
```

The third answer is the retrieval miss from the eval. The hours never reached
the top-5, so Claude refused instead of guessing a plausible "9–5." The failure
is visible and safe, not a confident hallucination.

### Engineering notes

- **Boilerplate removal mattered.** Every page repeated the same promo banner and
  form messages. Left in, a question like "what's the coupon code?" matches all 79
  pages and inflates hit@5. `crawl.py` drops any line that appears on 30%+ of pages
  before indexing.
- The embedding model is loaded once per process (`lru_cache`), not once per query.

## Design notes

- **Grounding over fluency:** the answer prompt forbids outside knowledge and
  requires citations; "Not found in the crawled content" is a success case,
  not a failure — in a compliance-adjacent product, a confident wrong answer
  is worse than an honest miss.
- **Local-first:** embeddings and the vector store run locally (no data leaves
  the machine until the final grounded-answer call), matching how the Answer
  Engine handles client site data.
- **Two configs on purpose:** chunk size is the highest-leverage retrieval
  knob; measuring both beats guessing.

Part of the Answer Engine family. Related: [flexo](https://github.com/Jonhutton/flexo) ·
[geo-prospecting](https://github.com/Jonhutton/geo-prospecting)
