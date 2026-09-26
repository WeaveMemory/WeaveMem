from __future__ import annotations


def calculate_answer_metrics(reference: str, prediction: str) -> dict[str, float]:
    """Compute token-set F1 and smoothed sentence BLEU-1 through BLEU-4."""
    try:
        import nltk
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError:
        raise RuntimeError(
            "Lexical metrics require NLTK; run `python -m pip install nltk==3.10.3`"
        ) from None

    try:
        reference_tokens = nltk.word_tokenize(str(reference).lower())
        prediction_tokens = nltk.word_tokenize(str(prediction).lower())
    except LookupError:
        raise RuntimeError(
            "NLTK tokenizer data is missing; run "
            "`python -m nltk.downloader punkt punkt_tab`"
        ) from None

    result = {"f1": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    if not reference_tokens or not prediction_tokens:
        return result

    reference_set = set(reference_tokens)
    prediction_set = set(prediction_tokens)
    overlap = len(reference_set & prediction_set)
    precision = overlap / len(prediction_set)
    recall = overlap / len(reference_set)
    if precision + recall:
        result["f1"] = float(2 * precision * recall / (precision + recall))

    smoothing = SmoothingFunction().method1
    weights = (
        (1, 0, 0, 0),
        (0.5, 0.5, 0, 0),
        (0.33, 0.33, 0.33, 0),
        (0.25, 0.25, 0.25, 0.25),
    )
    for index, weight in enumerate(weights, 1):
        result[f"bleu{index}"] = float(
            sentence_bleu(
                [reference_tokens], prediction_tokens, weights=weight,
                smoothing_function=smoothing,
            )
        )
    return result
