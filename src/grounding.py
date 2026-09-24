import re

from source_package import source_document_for_location


NUMERIC_FACT_PATTERN = re.compile(
    r"(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?"
    r"(?:\s*(?:-|to)\s*\d+(?:,\d{3})*(?:\.\d+)?)?"
    r"(?:\s*(?:%|[A-Za-z]+(?:[/-][A-Za-z]+)*))?",
    re.IGNORECASE,
)


def normalize_source_text(text):
    normalized, _ = normalize_source_with_index_map(text)
    return normalized


def normalize_source_with_index_map(text):
    replacements = {
        chr(8211): "-",
        chr(8212): "-",
        chr(8216): "'",
        chr(8217): "'",
        chr(8220): '"',
        chr(8221): '"',
    }
    normalized = []
    source_indexes = []
    pending_space_index = None

    for source_index, character in enumerate(str(text)):
        transformed = replacements.get(character, character).casefold()
        if transformed.isspace():
            if normalized and pending_space_index is None:
                pending_space_index = source_index
            continue
        if pending_space_index is not None:
            normalized.append(" ")
            source_indexes.append(pending_space_index)
            pending_space_index = None
        for output_character in transformed:
            normalized.append(output_character)
            source_indexes.append(source_index)

    return "".join(normalized), source_indexes


def source_contains_quote(source_text, quote):
    return bool(find_source_quote_locations(source_text, quote))


def find_source_quote_locations(source_text, quote):
    source_text = str(source_text)
    normalized_source, source_indexes = normalize_source_with_index_map(source_text)
    normalized_quote = normalize_source_text(quote)
    if not normalized_quote:
        return []

    locations = []
    search_start = 0
    while True:
        match_start = normalized_source.find(normalized_quote, search_start)
        if match_start < 0:
            break
        match_end = match_start + len(normalized_quote)
        source_start = source_indexes[match_start]
        source_end = source_indexes[match_end - 1] + 1
        start_line = source_text.count("\n", 0, source_start) + 1
        end_line = source_text.count("\n", 0, source_end - 1) + 1
        line_start_index = source_text.rfind("\n", 0, source_start) + 1
        locations.append({
            "start_char": source_start,
            "end_char": source_end,
            "start_line": start_line,
            "end_line": end_line,
            "start_column": source_start - line_start_index + 1,
        })
        search_start = match_start + 1
    return locations


def format_source_location(location, source_documents=None):
    source_document = source_document_for_location(
        location,
        source_documents or [],
    )
    if source_document:
        location = {
            **location,
            "start_char": location["start_char"] - source_document["start_char"],
            "end_char": location["end_char"] - source_document["start_char"],
            "start_line": location["start_line"] - source_document["start_line"] + 1,
            "end_line": location["end_line"] - source_document["start_line"] + 1,
        }
    line_label = (
        f"line {location['start_line']}"
        if location["start_line"] == location["end_line"]
        else f"lines {location['start_line']}-{location['end_line']}"
    )
    formatted = (
        f"{line_label}, column {location['start_column']}, "
        f"characters {location['start_char']}-{location['end_char']}"
    )
    if source_document:
        return f"{source_document['id']} ({source_document['name']}), {formatted}"
    return formatted


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
