# NLP Automated Customer Reviews

An end-to-end Natural Language Processing project for analyzing Amazon
customer reviews. The project combines sentiment classification, product
clustering, grounded LLM-generated category articles, and an interactive
sentiment-analysis application.

## Project Overview

Customer reviews contain useful information about satisfaction, product
quality, and recurring themes, but manually analyzing thousands of
reviews is difficult. This project builds an NLP pipeline that turns a
large collection of Amazon product reviews into structured insights.

The project has four main tasks:

1.  **Sentiment Analysis** --- classify reviews as Negative, Neutral, or
    Positive.
2.  **Product Clustering** --- group individual products into broader
    product meta-categories.
3.  **Generative AI** --- create short evidence-based category articles
    from computed review facts.
4.  **Deployment** --- make the sentiment classifier available through
    an interactive application.

An additional **Nemotron 3 Ultra** experiment evaluates
large-language-model sentiment classification through NVIDIA's hosted
API.

## Dataset

The project uses an Amazon product reviews dataset containing
approximately **34.6K cleaned reviews** across roughly 40 products.

Sentiment labels are derived from review ratings:

-   **1--2 stars:** Negative
-   **3 stars:** Neutral
-   **4--5 stars:** Positive

The dataset is strongly imbalanced toward positive reviews. This makes
accuracy alone misleading, so **Macro F1** is used as an important
evaluation metric.

## Task 1 --- Sentiment Classification

### TF-IDF + Logistic Regression

The traditional baseline uses TF-IDF text features and Logistic
Regression. Model settings were tuned using a validation split.

  Model                            Accuracy   Macro F1
  ------------------------------ ---------- ----------
  TF-IDF + Logistic Regression        93.6%      0.654

### Fine-tuned RoBERTa

`roberta-base` was fine-tuned for three-class sentiment classification.

  Model                            Negative F1   Neutral F1   Positive F1    Accuracy    Macro F1
  ------------------------------ ------------- ------------ ------------- ----------- -----------
  TF-IDF + Logistic Regression           0.608        0.385         0.969       93.6%       0.654
  RoBERTa                                0.717        0.435         0.976   **94.6%**   **0.709**

Neutral reviews remain the most difficult class. An always-positive
classifier can achieve high accuracy on this imbalanced dataset while
performing poorly across the three classes.

The large RoBERTa weight file is intentionally not stored in this GitHub
repository. Model hosting is handled separately through Hugging Face.

### Nemotron 3 Ultra Extension

The project also includes an experimental zero-shot sentiment evaluation
using NVIDIA Nemotron 3 Ultra through NVIDIA's hosted API.

A balanced diagnostic sample of 300 reviews was used: 100 Negative, 100
Neutral, and 100 Positive.

  Metric          Result
  ------------- --------
  Accuracy        0.7467
  Macro F1        0.7244
  Negative F1     0.8111
  Neutral F1      0.5325
  Positive F1     0.8297

**Important:** these Nemotron results are from a separate balanced
diagnostic sample and therefore should not be interpreted as a direct
performance comparison with the RoBERTa/TF-IDF test-set results.

Nemotron is accessed through an API; its model weights are not included
in this repository.

## Task 2 --- Product Clustering

Reviews are aggregated at the product level and products are represented
using textual/category information. Clustering groups individual Amazon
products into broader, interpretable categories.

  Meta-category     Products   Reviews   Avg. Rating
  --------------- ---------- --------- -------------
  Fire Tablets            13    17,605          4.49
  Echo                     3     7,261          4.66
  Fire TV                  4     5,080          4.70
  Kindle                   6     4,092          4.75
  Accessories             10       555          4.29

An important data-quality step was manually validating product names
because some raw product names were inaccurate or ambiguous.

## Task 3 --- Grounded Category Articles

The Generative AI pipeline follows a **fact-first** approach:

**Reviews → computed facts → structured prompt → LLM article →
validation**

Instead of asking the language model to infer statistics, Python
computes facts such as ratings, review counts, themes, complaints, and
praise first. The language model then turns those facts into readable
category-level articles.

Qwen was used as the main article-writing model. Prompt iterations
addressed issues such as invented product names or features, unsupported
claims, article truncation, misleading best/worst comparisons, and
numerical inconsistency.

> **Code computes the facts; the language model writes the narrative.**

Additional article-generation experiments are stored under the Review
Explorer application directory.

## Task 4 --- Deployment

The project includes an interactive sentiment-review application built
around the fine-tuned RoBERTa classifier.

Application source files are located in:

`apps/Sentiment Review/`

The project also contains a **Review Explorer** application for
exploring category-level results and generated articles:

`apps/Review Explorer/`

## Repository Structure

``` text
NLP_Customer_Reviews/
├── data/
│   ├── reviews_clean.csv
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── test_predictions.csv
│   ├── products_clustered.csv
│   ├── reviews_clustered.csv
│   └── roberta_best/
│       ├── config.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       └── training_args.bin
├── notebooks/
│   ├── 00_data_preparation_Roberto.ipynb
│   ├── 01_tfidf_baseline.ipynb
│   ├── 01_roberta_finetuning.ipynb
│   ├── 02_product_clustering.ipynb
│   ├── 03_category_articles.ipynb
│   ├── 04_deployment.ipynb
│   ├── nemotron_sentiment_mahmooda.ipynb
│   └── 06_llm_sentiment_api.ipynb
├── apps/
│   ├── Review Explorer/
│   └── Sentiment Review/
├── presentation/
│   ├── Customer Reviews NLP Workshop.pptx
│   └── Customer_Reviews_NLP_8min_Interactive.pptx
├── .gitignore
└── README.md
```

## Running the Project

The notebooks are designed to be run in Google Colab.

1.  Run `00_data_preparation_Roberto.ipynb` to prepare the dataset.
2.  Run `01_tfidf_baseline.ipynb` for the traditional sentiment
    baseline.
3.  Run `01_roberta_finetuning.ipynb` for transformer fine-tuning.
4.  Run `02_product_clustering.ipynb` for product-level clustering.
5.  Run `03_category_articles.ipynb` for grounded article generation.
6.  Use the deployment/API notebooks for the corresponding extensions.

Application-specific requirements are provided in their
`requirements.txt` files.

## Large Models and Secrets

Large model files are deliberately excluded from GitHub. In particular,
`data/roberta_best/model.safetensors` is ignored because it is too large
for normal GitHub storage.

API credentials must never be committed. Store NVIDIA, Hugging Face,
GitHub, and other credentials using environment variables, Google Colab
Secrets, or the hosting platform's secret-management feature.

## Key Findings

-   **Accuracy can hide class imbalance.** Macro F1 gives a more
    informative view across Negative, Neutral, and Positive reviews.
-   **Data quality matters before modeling.** Product names and category
    information require validation before downstream use.
-   **Generative AI needs grounding.** Computing facts in code before
    asking an LLM to write reduces numerical errors and unsupported
    claims.

## Contributors

Collaborative NLP project by **Mahmooda and Roberto Vargas**.

## License

See the repository license for reuse terms.
