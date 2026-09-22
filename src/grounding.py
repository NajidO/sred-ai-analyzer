import re


NUMERIC_FACT_PATTERN = re.compile(
    r"(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?"
    r"(?:\s*(?:-|to)\s*\d+(?:,\d{3})*(?:\.\d+)?)?"
    r"(?:\s*(?:%|[A-Za-z]+(?:[/-][A-Za-z]+)*))?",
    re.IGNORECASE,
)


def normalize_source_text(text):
    normalized = (
        text.replace(chr(8211), "-")
        .replace(chr(8212), "-")
        .replace(chr(8216), "'")
        .replace(chr(8217), "'")
        .replace(chr(8220), '"')
        .replace(chr(8221), '"')
    )
    return " ".join(normalized.split()).casefold()


def source_contains_quote(source_text, quote):
    normalized_quote = normalize_source_text(quote)
    return bool(normalized_quote) and normalized_quote in normalize_source_text(source_text)


def extract_numeric_facts(text):
    facts = set()
    for match in NUMERIC_FACT_PATTERN.finditer(normalize_source_text(text)):
        value = normalize_numeric_fact(match.group(0))
        if value:
            facts.add(value)
    return facts


def find_unsupported_numeric_facts(source_text, generated_text):
    source_facts = extract_numeric_facts(source_text)
    generated_facts = extract_numeric_facts(generated_text)
    return sorted(generated_facts - source_facts)


def normalize_numeric_fact(value):
    normalized = " ".join(value.casefold().split())
    normalized = normalized.replace(",", "")
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    return normalized.strip()


def unique_items(items):
    seen = set()
    result = []
    for item in items:
        marker = normalize_source_text(str(item))
        if marker and marker not in seen:
            result.append(item)
            seen.add(marker)
    return result
