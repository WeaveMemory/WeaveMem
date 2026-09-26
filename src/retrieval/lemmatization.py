

from __future__ import annotations

from retrieval.spacy_models import get_nlp_lemma


def lemmatize_for_bm25(text: str) -> str:
    source = text or ""
    nlp = get_nlp_lemma()
    if nlp is None:
        return source

    doc = nlp(source.lower())
    tokens: list[str] = []
    for token in doc:
        if token.is_punct or token.is_stop:
            continue
        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)
    return " ".join(tokens)
