"""Gradio app: classify an Amazon product review as negative, neutral or positive."""
import os

import gradio as gr
from transformers import pipeline

# The fine-tuned model on the Hugging Face Hub. Edit this, or set a MODEL_ID variable in the Space settings.
MODEL_ID = os.environ.get("MODEL_ID", "Roberto-Vargas/roberta-amazon-sentiment")
MAX_LENGTH = 128   # the model was trained with 128 tokens

# Safety net in case the model config only has generic label names
RENAME = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

_classifier = None


def get_classifier():
    """Load the model on the first request, so the app starts quickly."""
    global _classifier
    if _classifier is None:
        _classifier = pipeline("text-classification", model=MODEL_ID, top_k=None,
                               truncation=True, max_length=MAX_LENGTH)
    return _classifier


def to_scores(output):
    """Turn the pipeline output into {label: probability}. It is a list of dicts, sometimes wrapped in another list."""
    if output and isinstance(output[0], list):
        output = output[0]
    return {RENAME.get(d["label"], d["label"]): float(d["score"]) for d in output}


def classify(review):
    review = (review or "").strip()
    if not review:
        raise gr.Error("Please type or paste a review first.")
    return to_scores(get_classifier()(review))


EXAMPLES = [
    ["Love this tablet. The screen is bright, it was cheap, and my kids use it every day."],
    ["It stopped working after two weeks and customer service never answered my emails."],
    ["It's okay. Does what it says, but the battery could be better and the ads are annoying."],
    ["Alexa understands me most of the time. Sound is good for the size, setup was easy."],
    ["Terrible. Slow, freezes constantly, and I can't install the apps I need."],
]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=5, label="Customer review", placeholder="Paste an Amazon product review here..."),
    outputs=gr.Label(num_top_classes=3, label="Predicted sentiment"),
    examples=EXAMPLES,
    title="Amazon review sentiment",
    description=("Classifies a product review as **negative**, **neutral** or **positive**. "
                 "Fine-tuned RoBERTa trained on about 24,000 Amazon device reviews. "
                 "Neutral (3-star) reviews are the hardest class, so treat those predictions with care."),
)

if __name__ == "__main__":
    demo.launch()
