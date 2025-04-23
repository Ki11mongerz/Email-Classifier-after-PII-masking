import re
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

nlp = spacy.load("en_core_web_sm")

def mask_pii(text):
    masked_text = text
    entities = []

    # Define regex patterns in order of priority
    patterns = [
        ("dob", r'\b\d{2}/\d{2}/\d{4}\b'),  # Process DOB first to avoid conflicts
        ("credit_debit_no", r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'),
        ("phone_number", r'\+?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}\b'),
        ("email", r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        ("cvv_no", r'\b\d{3,4}\b'),
        ("expiry_no", r'\b\d{2}/\d{2}\b'),
    ]

    # First pass: Mask PII based on regex patterns in priority order
    for entity_type, pattern in patterns:
        matches = list(re.finditer(pattern, masked_text))
        for match in reversed(matches):  # Process matches in reverse order
            start, end = match.span()
            print(f"Regex Match: {match.group()} at [{start}, {end}]")
            masked_text = masked_text[:start] + f"[{entity_type}]" + masked_text[end:]
            entities.append({
                "position": [start, end],
                "classification": entity_type,
                "entity": match.group()
            })

    # Second pass: Mask full names using SpaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            start, end = ent.start_char, ent.end_char  # Use original text indices
            print(f"NER Match: {ent.text} at [{start}, {end}]")
            entities.append({
                "position": [start, end],
                "classification": "full_name",
                "entity": ent.text
            })
            # Update masked_text based on original indices
            masked_text = masked_text[:start] + "[full_name]" + masked_text[end:]

    # Ensure entities are sorted by their start position
    entities = sorted(entities, key=lambda x: x["position"][0])

    return masked_text, entities