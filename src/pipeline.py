from src.classifier import classify_medical_advice
from src.entity_detection import extract_medical_entities
from src.explanation import generate_explanation


danger_words = [
    "stop medicine",
    "avoid doctor",
    "replace treatment",
    "ignore doctor",
    "no need for hospital",
    "throw away your medicine"
]


def detect_severity(text):

    text_lower = text.lower()

    for word in danger_words:
        if word in text_lower:
            return "High Risk Advice"

    return "Normal Risk"


def detect_fake_medical_advice(text):

    # 1. classification
    label, confidence = classify_medical_advice(text)

    # 2. medical entity extraction
    entities = extract_medical_entities(text)

    # 3. explanation generation
    explanation = generate_explanation(text, label)

    # 4. risk detection
    severity = detect_severity(text)

    result = {
        "text": text,
        "classification": label,
        "confidence": round(confidence, 3),
        "entities": entities,
        "severity": severity,
        "explanation": explanation
    }

    return result

