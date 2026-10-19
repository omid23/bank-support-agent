"""Hybrid search over the knowledge base.

* BM25 over sections, with the document title prepended to every section, a light stemmer and
  stop-word removal; a document scores as its best section plus a share of its whole-text score.
* Dense vectors (LSA: TF-IDF + SVD) so paraphrases match ("money back" ~ "refund").
* Reciprocal rank fusion of the two.

Everything runs offline in milliseconds. The BM25 that tau2-bench ships (``rank_bm25`` over raw
content, ``str.split`` tokens) is reproduced in ``Tau2BM25`` as the baseline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .store import KnowledgeBase

STOPWORDS = set("""a an the and or but if then else of to in on at by for with from into over under about as is are was were be
been being am do does did doing have has had having i me my we our you your he she it its they them their this that these
those there here what which who whom whose when where why how all any both each few more most other some such no nor not
only own same so than too very can will just should would could may might must shall also please hi hello thanks thank
want wants wanted like would need needs get got im i'm i've i'd i'll you're it's dont don't""".split())

_SUFFIXES = ("ational", "ization", "fulness", "ousness", "iveness", "ments", "ment", "ings", "ing", "edly", "ies",
             "ied", "ers", "er", "ed", "es", "s")


def stem(word: str) -> str:
    """A small suffix stripper: disputes/disputed and charges/charged meet (a bare form ending in -e does not)."""
    if len(word) <= 4 or word.isdigit():
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            if suf == "ies" or suf == "ied":
                return base + "y"
            if suf in ("es", "ed", "er", "ers", "ing", "ings") and base.endswith("e") is False and len(base) > 3 \
                    and base[-1] == base[-2] and base[-1] not in "lsz":
                return base[:-1]
            return base
    return word


TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    out = []
    for t in TOKEN.findall(text.lower().replace("‑", "-")):
        for part in t.replace("'", "").split("-"):
            if part and part not in STOPWORDS:
                out.append(stem(part))
    return out


class BM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(corpus)
        self.lengths = np.array([len(d) for d in corpus], dtype=float)
        self.avgdl = float(self.lengths.mean()) if self.n else 0.0
        df: Counter = Counter()
        self.tfs: list[Counter] = []
        for d in corpus:
            tf = Counter(d)
            self.tfs.append(tf)
            df.update(tf.keys())
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, tf in enumerate(self.tfs):
            for t, c in tf.items():
                self.postings.setdefault(t, []).append((i, c))

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / (self.avgdl or 1))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            w = query.count(t) ** 0.5
            for i, c in self.postings[t]:
                out[i] += w * idf * c * (self.k1 + 1) / (c + norm[i])
        return out


@dataclass
class Hit:
    doc_id: str
    score: float
    title: str
    section: str = ""
    sources: dict[str, int] = field(default_factory=dict)


class Searcher:
    """Hybrid section-level search; returns documents."""

    def __init__(self, kb: KnowledgeBase, dense_dims: int = 128, weights: dict[str, float] | None = None,
                 doc_weight: float = 0.35):
        self.kb = kb
        self.weights = weights or {"bm25": 1.0, "dense": 1.0}
        self.doc_weight = doc_weight
        self.units = []  # (doc index, heading)
        texts = []
        for di, d in enumerate(kb.documents):
            for s in d.sections:
                self.units.append((di, s.heading))
                texts.append(f"{d.title} {d.title} {s.heading} {s.text}")
        self.unit_bm25 = BM25([tokenize(t) for t in texts])
        self.doc_bm25 = BM25([tokenize(f"{d.title} {d.title} {d.content}") for d in kb.documents])
        self.dense = _LSA([f"{d.title}. {d.content}" for d in kb.documents], dense_dims) if dense_dims else None

    def bm25_doc_scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        per_unit = self.unit_bm25.scores(q)
        best = np.zeros(len(self.kb.documents))
        for (di, _), s in zip(self.units, per_unit, strict=False):
            if s > best[di]:
                best[di] = s
        whole = self.doc_bm25.scores(q)
        return best + self.doc_weight * whole

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[Hit]:
        if not query.strip():
            return []
        rankings: dict[str, list[int]] = {}
        if mode in ("bm25", "hybrid"):
            s = self.bm25_doc_scores(query)
            rankings["bm25"] = [int(i) for i in np.argsort(-s)[:200] if s[i] > 0]
        if mode in ("dense", "hybrid") and self.dense is not None:
            rankings["dense"] = list(self.dense.rank(query, 200))
        fused: dict[int, float] = {}
        ranks: dict[int, dict[str, int]] = {}
        for name, ids in rankings.items():
            w = self.weights.get(name, 1.0) if mode == "hybrid" else 1.0
            for r, i in enumerate(ids, start=1):
                fused[i] = fused.get(i, 0.0) + w / (60 + r)
                ranks.setdefault(i, {})[name] = r
        order = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.kb.documents[i].id, round(fused[i], 6), self.kb.documents[i].title, sources=ranks[i])
                for i in order]


class _LSA:
    def __init__(self, texts: list[str], dims: int):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 2), min_df=1, max_df=0.6,
                                   sublinear_tf=True, token_pattern=None)
        x = self.vec.fit_transform(texts)
        dims = max(2, min(dims, x.shape[1] - 1, x.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=dims, random_state=0)
        m = self.svd.fit_transform(x)
        self.matrix = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)

    def rank(self, query: str, k: int) -> np.ndarray:
        q = self.svd.transform(self.vec.transform([query]))[0]
        q = q / max(np.linalg.norm(q), 1e-9)
        return np.argsort(-(self.matrix @ q))[:k]


class Tau2BM25:
    """The retriever tau2-bench's ``bm25`` config uses: rank_bm25.BM25Okapi over raw content, split on spaces."""

    def __init__(self, kb: KnowledgeBase):
        from rank_bm25 import BM25Okapi

        self.kb = kb
        self.bm25 = BM25Okapi([d.content.lower().split() for d in kb.documents])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        s = self.bm25.get_scores(query.lower().split())
        order = sorted(range(len(s)), key=lambda i: s[i], reverse=True)[:k]
        return [Hit(self.kb.documents[i].id, float(s[i]), self.kb.documents[i].title) for i in order]
