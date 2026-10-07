---
title: Amazon Review Sentiment
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Amazon review sentiment

A fine-tuned `roberta-base` model that classifies Amazon product reviews as negative, neutral or positive.

Trained on the Datafiniti *Consumer Reviews of Amazon Products* dataset (`1429_1.csv`), with star ratings mapped to
1-2 negative, 3 neutral and 4-5 positive. About 93% of the reviews are positive, so the model was trained with class
weights and evaluated mainly with macro-F1.

**Limitations:** the dataset is almost entirely Amazon devices (tablets, Kindles, Echo, Fire TV), so the model may
be less reliable on other product types. Neutral reviews are the hardest class.
