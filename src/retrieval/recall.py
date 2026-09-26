

def mid_text(mid: dict) -> str:
    """Represent a Mid using its topic and summary."""
    return f"{mid.get('topic_subject') or ''} {mid.get('summary') or ''}".strip()
