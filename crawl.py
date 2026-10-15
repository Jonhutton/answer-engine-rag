"""
Polite sitemap crawler that builds the RAG corpus.

Usage (PowerShell):
  python crawl.py --site https://www.example-business.com --out crawled_pages --max 60

Reads the site's sitemap.xml, honors robots.txt, skips account/auth/search pages,
waits between requests, and saves each page as clean text (nav, footer, scripts
stripped) with its URL on the first line so citations point back to the live page.
"""

import argparse, pathlib, re, time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

UA = "AnswerEngineRAG/1.0 (+https://foundation-ai.net)"
SKIP = re.compile(r"/(account|wholesale-account|login|signup|cart|checkout|search)(/|$)", re.I)

def sitemap_urls(site: str, session: requests.Session) -> list[str]:
    seen, queue, urls = set(), [site.rstrip("/") + "/sitemap.xml"], []
    while queue:
        sm = queue.pop()
        if sm in seen:
            continue
        seen.add(sm)
        xml = session.get(sm, timeout=20).text
        locs = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml)
        if "<sitemapindex" in xml:
            queue.extend(locs)
        else:
            urls.extend(locs)
    return urls

def page_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "svg", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    lines = (ln.strip() for ln in main.get_text("\n").splitlines())
    return "\n".join(ln for ln in lines if ln)

def slug(url: str) -> str:
    path = urlparse(url).path.strip("/") or "home"
    return re.sub(r"[^a-z0-9]+", "_", path.lower()).strip("_")[:120]

def strip_boilerplate(folder: pathlib.Path, threshold: float = 0.3):
    """Drop lines that repeat across many pages (promo banners, form messages, schema
    metadata). Left in, they'd make every page 'match' questions about those lines and
    inflate retrieval scores."""
    files = sorted(folder.glob("*.txt"))
    pages = {p: p.read_text(encoding="utf-8").splitlines() for p in files}
    df: dict[str, int] = {}
    for lines in pages.values():
        for ln in set(lines[1:]):
            df[ln] = df.get(ln, 0) + 1
    cutoff = max(3, int(len(files) * threshold))
    common = {ln for ln, n in df.items() if n >= cutoff}
    for p, lines in pages.items():
        kept = [lines[0]] + [ln for ln in lines[1:] if ln not in common
                             and not ln.startswith(("http://", "https://"))]
        p.write_text("\n".join(kept), encoding="utf-8")
    print(f"Removed {len(common)} boilerplate lines repeated on {cutoff}+ of {len(files)} pages")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site")
    ap.add_argument("--out", default="crawled_pages")
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--clean-only", action="store_true", help="only re-run boilerplate removal")
    args = ap.parse_args()
    if args.clean_only:
        return strip_boilerplate(pathlib.Path(args.out))
    if not args.site:
        ap.error("--site is required")

    s = requests.Session()
    s.headers["User-Agent"] = UA
    robots = RobotFileParser(args.site.rstrip("/") + "/robots.txt")
    robots.read()

    urls = [u for u in sitemap_urls(args.site, s)
            if robots.can_fetch(UA, u) and not SKIP.search(urlparse(u).path)]
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)
    print(f"{len(urls)} crawlable URLs in sitemap; fetching up to {args.max}")

    saved = 0
    for url in urls[: args.max]:
        try:
            r = s.get(url, timeout=20)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  skip {url}: {e}")
            continue
        text = page_text(r.text)
        if len(text.split()) < 30:
            print(f"  thin {url}")
            continue
        (out / f"{slug(url)}.txt").write_text(f"URL: {url}\n\n{text}", encoding="utf-8")
        saved += 1
        print(f"  ok   {url} ({len(text.split())} words)")
        time.sleep(args.delay)
    print(f"Saved {saved} pages to {out}/")
    strip_boilerplate(out)

if __name__ == "__main__":
    main()
