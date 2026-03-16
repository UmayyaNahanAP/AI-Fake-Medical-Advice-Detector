from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

candidate_labels = [
    "medical misinformation",
    "medically correct advice"
]


def classify_medical_advice(text):

    hypothesis_template = "This statement is {}."

    result = classifier(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template=hypothesis_template
    )

    top_label = result["labels"][0]
    confidence = result["scores"][0]

    if top_label == "medical misinformation":
        label = "misinformation"
    else:
        label = "correct"

    return label, confidence



