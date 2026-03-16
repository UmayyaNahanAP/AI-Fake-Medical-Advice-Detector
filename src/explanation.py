from transformers import pipeline

# text generation model
generator = pipeline(
    "text-generation",
    model="gpt2"
)

def generate_explanation(text, label):

    prompt = f"""
Health advice: {text}

Classification: {label}

Explain briefly why this advice is {label}.
Explanation:
"""

    result = generator(
        prompt,
        max_new_tokens=60,
        do_sample=True
    )

    explanation = result[0]["generated_text"]

    return explanation
