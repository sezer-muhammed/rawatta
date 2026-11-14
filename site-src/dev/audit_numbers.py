# SPDX-License-Identifier: Apache-2.0
"""List every number in the visible text (and alt text) of docs/index.html and find it in the paper's LaTeX sources.

The paper sources are not part of this repository; pass the directory that holds main.tex and sec/*.tex.
Output (TSV on stdout): section, number, match, page context, best tex match (file:line), tex line excerpt.
  match = "context"    the number occurs on a source line that shares at least two words with the page context
          "weak"       the number occurs in the sources, but not with matching words (common numbers such as 0, 1,
                       10, 50 or 200 nearly always land here): check it by hand
          "NOT FOUND"  the number does not occur in the sources
Finding a number is necessary, not sufficient: a number quoted in the wrong context still matches somewhere. The
hand-checked mapping of every page number to its source is kept by the authors with the paper sources; this script
is a first pass for it.
Identifiers (section/table/figure references, metric names, versions, licences, CLI flags) are skipped.
"""
import argparse, html.parser, os, re, sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("paper_dir", help="the paper's LaTeX source directory (holds main.tex and sec/*.tex)")
ap.add_argument("page", nargs="?", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs", "index.html"),
                help="page to audit (default: docs/index.html)")
args = ap.parse_args()
paper, page = args.paper_dir, args.page
if not os.path.isfile(os.path.join(paper, "main.tex")):
    ap.error(f"{paper}: no main.tex here (pass the paper's LaTeX source directory)")
if not os.path.isfile(page):
    ap.error(f"{page}: not found")
SKIP_TAGS = {"script", "style", "head"}


class Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(); self.stack = []; self.section = "header"; self.chunks = []; self.cur = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "section" and a.get("id"): self.flush(); self.section = a["id"]
        if tag == "footer": self.flush(); self.section = "footer"
        if tag == "img" and a.get("alt"): self.chunks.append((self.section, "[alt] " + a["alt"]))
        if tag in ("p", "li", "tr", "figcaption", "h1", "h2", "h3", "pre", "caption", "div", "article"): self.flush()
        if tag in ("td", "th", "br"): self.cur.append(" | ")
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in ("p", "li", "tr", "figcaption", "h1", "h2", "h3", "pre", "caption", "article"): self.flush()
        while self.stack and self.stack.pop() != tag: pass

    def handle_data(self, d):
        if not any(t in SKIP_TAGS for t in self.stack): self.cur.append(d)

    def flush(self):
        t = " ".join("".join(self.cur).split())
        if t: self.chunks.append((self.section, t))
        self.cur = []


p = Text(); p.feed(open(page, encoding="utf-8").read()); p.flush()

tex = []
for root, _, files in os.walk(paper):
    for f in sorted(files):
        if f.endswith((".tex", ".bib")):
            path = os.path.join(root, f)
            for i, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
                tex.append((os.path.relpath(path, paper), i, line.rstrip("\n")))

# identifiers that are not results
IDENT = re.compile(r"(Tab\.|Sec\.|Fig\.|Figure|Table|Sections?)\s*[\dA-Z][\d.]*|AUC@\d+|Δ\s*\d+|sm_\d+|compute_\d+|three@[\d.]+|x86_64|"
                   r"CC BY \d\.\d|Apache-2\.0|\b(?:Rust|CUDA) \d+\.\d+(?= or newer)|vggt_omega_\w+|\$\(id[^)]*\)|XXXX\.XXXXX|sezer2026rawatta|"
                   r"Paper \(arXiv\)|[A-Za-z]+\d+[A-Za-z]*|\d+[A-Za-z_]+\w*|δ\s*1\.25")
NUM = re.compile(r"(?<![\w.])[−-]?\d+(?:[.,]\d+)?(?![\w])")


def words(s): return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z\-]{3,}", s)}


def norm(n): return n.replace("−", "-").replace(",", "").lstrip("-")


print("section\tnumber\tmatch\tpage context\tsource\ttex excerpt")
counts = {"context": 0, "weak": 0, "NOT FOUND": 0}
for section, text in p.chunks:
    clean = IDENT.sub(" ", text)
    for m in NUM.finditer(clean):
        n = norm(m.group())
        ctx = clean[max(0, m.start() - 60): m.end() + 40].strip()
        pat = re.compile(r"(?<![\d.])" + re.escape(n) + r"(?![\d])")
        hits = [(f, i, l) for f, i, l in tex if pat.search(l.replace("{,}", "").replace("\\,", ""))]
        if not hits:
            counts["NOT FOUND"] += 1
            print(f"{section}\t{n}\tNOT FOUND\t{ctx}\t\t"); continue
        cw = words(ctx)
        best = max(hits, key=lambda h: (len(cw & words(h[2])), -h[1] if h[0].startswith("sec/") else -10 ** 6 - h[1]))
        match = "context" if len(cw & words(best[2])) >= 2 else "weak"
        counts[match] += 1
        k = pat.search(best[2]).start()
        excerpt = best[2][max(0, k - 70): k + 50].strip()
        print(f"{section}\t{n}\t{match}\t{ctx}\t{best[0]}:{best[1]}\t{excerpt}")
print(f"numbers: {sum(counts.values())}; with context {counts['context']}, weak {counts['weak']}, not found {counts['NOT FOUND']}"
      " (weak and not-found ones need a hand check)", file=sys.stderr)
