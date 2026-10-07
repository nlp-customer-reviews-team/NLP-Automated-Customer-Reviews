---
title: Review Explorer
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Review Explorer

Explore Amazon device reviews by product category and click a product to read its review summary: ratings, what
reviewers like, what they complain about, and real example reviews.

- **Quick summary:** built from exact numbers in the data, no model needed.
- **AI summary (optional):** a small Qwen model rewrites the facts as prose, and the numbers are checked afterwards.
- **Sentiment model check (optional):** set the `MODEL_ID` variable to a fine-tuned sentiment model to compare its
  predictions with the star ratings.

Optional files next to `app.py`: `category_facts.json`, `category_articles.md` and `prompt_evaluation.csv`, exported by the Task 3 notebook. If present, the app shows your generated guides and prompt scores and uses your notebook's numbers.

Data: Datafiniti *Consumer Reviews of Amazon Products* (`1429_1.csv`), processed in the Task 1-3 notebooks.
