"""Review Explorer: pick a product category, click a product, read its review summary.

Data comes from the Task 1-3 pipeline (data.json). Everything works with no model:
  - the quick summary is built from exact numbers, so it cannot make anything up;
  - optional extras: an AI-written summary (Qwen) and a check of the sentiment model on real reviews.
"""
import json
import os
import re

import gradio as gr

try:                                    # ZeroGPU Spaces need a function marked with @spaces.GPU
    import spaces
    gpu = spaces.GPU
except ImportError:                     # normal hardware: the decorator does nothing
    def gpu(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda fn: fn

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(os.path.join(HERE, "data.json"), encoding="utf-8"))
CATS = {c["name"]: c for c in DATA["categories"]}
CAT_NAMES = [c["name"] for c in DATA["categories"]]

# ------------------------------------------------------------------ the notebook's export files (optional, next to app.py)
#   category_facts.json   -> numbers for the top and worst products (overrides the built-in ones)
#   category_articles.md  -> the generated guides, in the same order as category_facts.json
#   prompt_evaluation.csv -> the prompt comparison scores
def _read(name):
    path = os.path.join(HERE, name)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else None


def _load_json(name):
    raw = _read(name)
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        return {}


NB_FACTS = _load_json("category_facts.json")
NB_ARTICLES_RAW = _read("category_articles.md") or ""


def map_names(names):
    """Match the notebook's category names to this app's. If a name differs, fall back to position (biggest category first)."""
    out = {}
    for i, n in enumerate(names):
        if n in CATS:
            out[n] = n
        elif i < len(CAT_NAMES):
            out[n] = CAT_NAMES[i]
    return out


FACT_KEYS = map_names(list(NB_FACTS))


def fact_text(p):
    """Fact sheet for one product, in the same layout as the notebook (used by the AI summary)."""
    lines = [f"{p['name']}: {p['n']} reviews, average {p['avg']} out of 5, {p['pctPos']}% positive (4-5 stars), {p['pctNeg']:g}% negative (1-2 stars)."]
    if p.get("mentions"):
        lines.append("   Topics reviewers mention more often for this product than for the others here (topics only, not praise or complaints): "
                     + ", ".join(p["mentions"]) + ".")
    if p["praise"]:
        lines.append("   What reviewers praise (share of its 4-5 star reviews): " + "; ".join(f"{l} ({s}%)" for l, s in p["praise"]) + ".")
    else:
        lines.append("   No praise theme stands out in its positive reviews.")
    if p["complaints"]:
        lines.append(f"   Top complaints, as a share of its {p['negCut']}-star reviews ({p['nNeg']} reviews): "
                     + "; ".join(f"{l} ({s}%)" for l, s in p["complaints"]) + ".")
        lines += [f'   Example complaint: "{q}"' for q in p["complaintQuotes"]]
    else:
        lines.append("   No complaint theme stands out in its negative reviews.")
    return "\n".join(lines)


def apply_notebook_facts():
    """Use the notebook's own numbers for its top and worst products. Returns how many products were updated."""
    updated = 0
    for key, f in NB_FACTS.items():
        cat = FACT_KEYS.get(key)
        if not cat:
            continue
        for e in list(f.get("top", [])) + [f.get("worst") or {}]:
            p = next((x for x in CATS[cat]["list"] if x["name"] == e.get("name")), None)
            if not p:
                continue
            p.update(n=int(e["reviews"]), pctPos=int(e["pct_pos"]), pctNeg=float(e["pct_neg"]), negCut=e["neg_cut"], nNeg=int(e["n_neg"]),
                     complaints=[[l, int(v)] for l, v in e["complaints"]], praise=[[l, int(v)] for l, v in e.get("praise", [])],
                     complaintQuotes=list(e.get("quotes", [])), mentions=list(e.get("mentions", [])))
            p["fact"] = fact_text(p)
            updated += 1
    return updated


def parse_articles():
    parts = [t.strip() for t in re.split(r"\n\s*-{3,}\s*\n", NB_ARTICLES_RAW) if t.strip()]
    keys = list(NB_FACTS) or CAT_NAMES                     # the notebook writes articles in the order of category_facts.json
    return {FACT_KEYS.get(keys[i], keys[i]): t for i, t in enumerate(parts) if i < len(keys)}


def load_eval():
    path = os.path.join(HERE, "prompt_evaluation.csv")
    if not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    mapping = map_names(list(dict.fromkeys(df["category"])))
    df["category"] = df["category"].map(lambda c: mapping.get(c, c))
    return df


PRODUCTS_UPDATED = apply_notebook_facts()
ARTICLES = parse_articles()
EVAL = load_eval()
SCORE_COLS = [c for c in ["numbers_real", "products_covered", "complaints_covered", "length_ok", "structure_ok", "names_ok"]
              if EVAL is not None and c in EVAL.columns]
BEST_VARIANT = (EVAL.groupby("variant")["overall"].mean().idxmax() if EVAL is not None and "overall" in EVAL.columns else None)

# Optional model settings (set them as variables in the Space settings)
MODEL_ID = os.environ.get("MODEL_ID", "")                               # your fine-tuned sentiment model, e.g. Roberto-Vargas/<repo>
AI_MODEL = os.environ.get("AI_MODEL", "Qwen/Qwen2.5-3B-Instruct")       # writes the AI summary
LABELS = ["negative", "neutral", "positive"]
STAR_COLORS = ["#C2410C", "#F08A3C", "#A3A8AE", "#8FB0FF", "#2457F5"]

TEXT, MUTED, RULE = "var(--body-text-color, #101418)", "var(--body-text-color-subdued, #5F666D)", "var(--border-color-primary, #E1E3E5)"


# ------------------------------------------------------------------ helpers
def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def products_of(cat_name):
    return CATS[cat_name]["list"]


def get_product(cat_name, product_name):
    for p in products_of(cat_name):
        if p["name"] == product_name:
            return p
    return products_of(cat_name)[0]


def table_rows(cat_name):
    return [[p["name"], p["n"], p["avg"], f'{p["pctNeg"]:g}%'] for p in products_of(cat_name)]


def joined(items):
    items = list(items)
    return ", ".join(items[:-1]) + " and " + items[-1] if len(items) > 1 else (items[0] if items else "")


# ------------------------------------------------------------------ quick summary (no model, exact numbers)
def quick_summary(cat_name, p):
    cat = CATS[cat_name]
    plist = products_of(cat_name)
    rank = sorted(plist, key=lambda x: -x["n"]).index(p) + 1
    diff = round(p["avg"] - cat["avg"], 2)
    where = "above" if diff > 0.05 else "below" if diff < -0.05 else "in line with"
    compare = (f"Its average rating ({p['avg']:.2f}) is {where} the category average ({cat['avg']:.2f})"
               if where != "in line with" else f"Its average rating ({p['avg']:.2f}) is in line with the category average ({cat['avg']:.2f})")
    lines = [
        f"### {p['name']}",
        f"**{p['n']:,} reviews · {p['avg']:.1f} out of 5 · {p['pctPos']}% positive · {p['pctNeg']:g}% negative**",
        "",
        f"It is the #{rank} most-reviewed of {len(plist)} ranked products in *{cat_name}*. {compare}.",
        "",
    ]
    lines.append("**What reviewers like:** " + (joined(f"{l} ({s}%)" for l, s in p["praise"]) + f" (share of its 4-5 star reviews)." if p["praise"]
                                                    else "no praise theme stands out."))
    lines.append("")
    if p["complaints"]:
        lines.append("**What they complain about:** " + joined(f"{l} ({s}%)" for l, s in p["complaints"]) +
                     f" (share of its {p['negCut']}-star reviews, {p['nNeg']} reviews).")
    else:
        lines.append("**What they complain about:** no complaint theme stands out.")
    if p["nNeg"] < 15:
        lines += ["", f"*Only {p['nNeg']} negative reviews, so the complaint shares are rough.*"]
    lines += ["", "*Themes are keyword matches over the reviews, so one review can count for several themes.*"]
    return "\n".join(lines)


# ------------------------------------------------------------------ HTML pieces
def rating_stack(ratings, height=24):
    tot = sum(ratings) or 1
    segs = "".join(
        f'<i style="display:block;flex:{n};background:{STAR_COLORS[i]}" title="{i + 1} star: {n:,} reviews ({round(n / tot * 100)}%)"></i>'
        for i, n in enumerate(ratings))
    legend = "".join(
        f'<span style="margin-right:14px;font-size:13px;color:{MUTED}"><i style="display:inline-block;width:11px;height:11px;border-radius:3px;'
        f'background:{STAR_COLORS[i]};margin-right:5px"></i>{i + 1} star{"s" if i else ""} {round(ratings[i] / tot * 100)}%</span>' for i in range(5))
    return (f'<div style="display:flex;height:{height}px;border-radius:6px;overflow:hidden;background:{RULE}">{segs}</div>'
            f'<div style="margin-top:8px">{legend}</div>')


def chips(items, color):
    if not items:
        return f'<span style="color:{MUTED};font-size:14px">No theme stands out</span>'
    return "".join(f'<span style="display:inline-block;margin:0 6px 6px 0;padding:3px 11px;border-radius:999px;border:1px solid {color};'
                   f'font-size:14px;color:{TEXT}">{esc(l)} <b>{s}%</b></span>' for l, s in items)


def vs_category(p, cat):
    """Two thin markers: this product's average rating next to its category's, on a 1-5 scale."""
    def pos(v):
        return max(0, min(100, (v - 1) / 4 * 100))
    return (f'<div style="position:relative;height:34px;margin:6px 0 2px">'
            f'<div style="position:absolute;top:15px;left:0;right:0;height:4px;border-radius:2px;background:{RULE}"></div>'
            f'<div style="position:absolute;top:8px;left:{pos(cat["avg"])}%;width:2px;height:18px;background:{MUTED}"></div>'
            f'<div style="position:absolute;top:6px;left:calc({pos(p["avg"])}% - 7px);width:14px;height:22px;border-radius:7px;background:#2457F5"></div></div>'
            f'<div style="font-size:13px;color:{MUTED}">Blue: this product ({p["avg"]:.2f}) · grey line: category average ({cat["avg"]:.2f}) · scale 1 to 5</div>')


def product_charts(cat_name, p):
    cat = CATS[cat_name]
    return (f'<div style="display:flex;flex-direction:column;gap:18px;color:{TEXT}">'
            f'<div><div style="font-weight:600;margin-bottom:8px">How people rated it</div>{rating_stack(p["ratings"])}</div>'
            f'<div><div style="font-weight:600;margin-bottom:2px">Against its category</div>{vs_category(p, cat)}</div>'
            f'<div><div style="font-weight:600;margin-bottom:8px">Reviewers like</div>{chips(p["praise"], "#2457F5")}</div>'
            f'<div><div style="font-weight:600;margin-bottom:8px">Reviewers complain about</div>{chips(p["complaints"], "#E8731A")}</div></div>')


def product_quotes(p):
    def block(title, items, color):
        if not items:
            return ""
        qs = "".join(f'<blockquote style="margin:0 0 10px;padding:10px 14px;border-left:4px solid {color};font-size:15px;color:{TEXT}">'
                     f'“{esc(q)}”</blockquote>' for q in items)
        return f'<div style="margin-bottom:6px;font-weight:600;color:{TEXT}">{title}</div>{qs}'
    body = block("In their words: praise", p["praiseQuotes"], "#2457F5") + block("In their words: complaints", p["complaintQuotes"], "#E8731A")
    note = f'<div style="font-size:13px;color:{MUTED}">One review each. They illustrate a theme and are not a trend on their own.</div>'
    return f"<div>{body}{note}</div>" if body else ""


def render_product(cat_name, product_name):
    p = get_product(cat_name, product_name)
    return quick_summary(cat_name, p), product_charts(cat_name, p), product_quotes(p), p["name"]


# ------------------------------------------------------------------ compare categories (second tab)
def themes_overall(k=8):
    best = {}
    for c in DATA["categories"]:
        for t, v in c["themes"].items():
            best[t] = max(best.get(t, 0), v)
    return [t for t, v in sorted(best.items(), key=lambda x: -x[1]) if v > 0][:k]


def compare_html():
    cats = DATA["categories"]
    rows = "".join(
        f'<div style="display:grid;grid-template-columns:minmax(120px,240px) 1fr;gap:6px 16px;align-items:center;margin-bottom:12px">'
        f'<div style="font-weight:500">{esc(c["name"])}<div style="font-size:13px;color:{MUTED}">{c["reviews"]:,} reviews · {c["avg"]:.1f} / 5</div></div>'
        f'<div>{rating_stack(c["ratings"], 20)}</div></div>' for c in cats)
    head = "<th></th>" + "".join(f'<th style="text-align:center;padding:6px 8px;font-size:13px;color:{MUTED}">{esc(re.sub(r" [(].*", "", c["name"]))}</th>' for c in cats)
    body = ""
    for t in themes_overall():
        cells = ""
        for c in cats:
            v = c["themes"].get(t, 0)
            a = min(55, round(v * 1.6))
            cells += (f'<td style="text-align:center;padding:9px 6px;border-radius:8px;color:{TEXT};'
                      f'background:color-mix(in srgb, #E8731A {a}%, transparent)">{f"{v}%" if v else "·"}</td>')
        body += f'<tr><td style="padding:6px 8px;font-weight:500;color:{TEXT}">{esc(t)}</td>{cells}</tr>'
    return (f'<div style="color:{TEXT}"><div style="font-weight:700;font-size:18px;margin-bottom:10px">How each category was rated</div>{rows}'
            f'<div style="font-weight:700;font-size:18px;margin:22px 0 4px">What customers complain about</div>'
            f'<div style="font-size:14px;color:{MUTED};margin-bottom:8px">Share of each category\'s negative reviews that mention a theme (keyword-based).</div>'
            f'<div style="overflow-x:auto"><table style="border-collapse:separate;border-spacing:3px;width:100%;min-width:560px;font-size:14px">'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>')


# ------------------------------------------------------------------ AI summary (optional; uses a language model)
SYSTEM_PROMPT = (
    "You are an editor at a consumer-electronics buying-guide website. You write honest, concrete summaries "
    "based ONLY on the review data you are given.\n"
    "Rules:\n"
    "- Use only facts that appear in the data. Never mention a product, model, feature, price or specification that is not in the data.\n"
    "- Praise a product only for what appears under 'What reviewers praise'. Describe problems only with the listed complaints.\n"
    "- Copy the product name exactly. Quote numbers exactly as given.\n"
    "- Each percentage belongs to the item it is listed next to. Never merge two items under one percentage.\n"
    "- Example complaints are single reviews: say 'one reviewer wrote' if you quote one.")
_llm = {}


def check_ai_text(text, fact):
    allowed = set(re.findall(r"\d+(?:\.\d+)?", fact)) | {"1", "2", "3", "4", "5"}
    nums = re.findall(r"\d+(?:\.\d+)?", re.sub(r"(\d),(?=\d{3})", r"\1", text))
    missing = sorted({n for n in nums if n not in allowed})
    return ("All numbers in the summary appear in the fact sheet." if not missing
            else "Numbers not found in the fact sheet: " + ", ".join(missing[:6]) + ". Check them against the quick summary.")


@gpu(duration=120)
def ai_summary(cat_name, product_name):
    if not product_name:
        return "Click a product in the table first."
    p = get_product(cat_name, product_name)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if "model" not in _llm:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            dt = torch.float16 if dev == "cuda" else torch.float32
            tok = AutoTokenizer.from_pretrained(AI_MODEL)
            try:
                mdl = AutoModelForCausalLM.from_pretrained(AI_MODEL, dtype=dt)
            except TypeError:                       # older transformers call it torch_dtype
                mdl = AutoModelForCausalLM.from_pretrained(AI_MODEL, torch_dtype=dt)
            _llm.update(tok=tok, model=mdl.to(dev).eval(), dev=dev)
        tok, mdl, dev = _llm["tok"], _llm["model"], _llm["dev"]
        user = (f"Write a customer-review summary of about 120 words for '{p['name']}' (category: {cat_name}). "
                "Two short paragraphs: what reviewers like, then what they complain about. End with one sentence of verdict. "
                "If the complaints are a small share of the reviews, say so plainly.\n\nDATA:\n" + p["fact"])
        text = tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = mdl.generate(**inputs, max_new_tokens=320, do_sample=True, temperature=0.2, top_p=0.9,
                               repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
        answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception as exc:                        # keep the app usable if the model cannot load
        return f"The AI summary could not run: {exc}\n\nThe quick summary above does not need a model."
    return f"{answer}\n\n---\n*Written by {AI_MODEL}. {check_ai_text(answer, p['fact'])}*"


# ------------------------------------------------------------------ sentiment model check (optional)
_clf = {}


def to_scores(output):
    if output and isinstance(output[0], list):
        output = output[0]
    rename = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    return {rename.get(d["label"], d["label"]): float(d["score"]) for d in output}


def star_label(r):
    return "negative" if r <= 2 else "neutral" if r == 3 else "positive"


def sentiment_check(cat_name, product_name):
    if not product_name:
        return "<p>Click a product in the table first.</p>"
    if not MODEL_ID:
        return ("<p>Set a <code>MODEL_ID</code> variable in the Space settings to the repo of your fine-tuned sentiment model "
                "(for example <code>Roberto-Vargas/your-model</code>) to compare it with the star ratings.</p>")
    p = get_product(cat_name, product_name)
    try:
        if "clf" not in _clf:
            from transformers import pipeline
            _clf["clf"] = pipeline("text-classification", model=MODEL_ID, top_k=None, truncation=True, max_length=128)
        outs = _clf["clf"]([s["text"] for s in p["samples"]], batch_size=8)
    except Exception as exc:
        return f"<p>The sentiment model could not run: {esc(exc)}</p>"
    rows, hits = "", 0
    for s, o in zip(p["samples"], outs):
        sc = to_scores(o)
        pred = max(sc, key=sc.get)
        want = star_label(s["rating"])
        ok = pred == want
        hits += ok
        text = esc(s["text"][:150] + ("..." if len(s["text"]) > 150 else ""))
        rows += (f'<tr><td style="padding:6px 8px">{text}</td><td style="padding:6px 8px;text-align:center">{s["rating"]}</td>'
                 f'<td style="padding:6px 8px">{want}</td><td style="padding:6px 8px">{pred} ({sc[pred] * 100:.0f}%)</td>'
                 f'<td style="padding:6px 8px;text-align:center;font-weight:700;color:{"#1B7F4B" if ok else "#B3560B"}">{"match" if ok else "differs"}</td></tr>')
    n = len(p["samples"])
    th = "".join(f'<th style="text-align:left;padding:6px 8px;font-size:13px;color:{MUTED}">{h}</th>' for h in ["Review", "Stars", "Stars say", "Model says", ""])
    return (f'<div style="color:{TEXT}"><p style="margin-bottom:8px"><b>The model agrees with the star rating on {hits} of {n} sampled reviews.</b> '
            f'The sample deliberately includes extra negative and neutral reviews, so this is not the overall accuracy. '
            f'Mixed reviews are where they differ most.</p><div style="overflow-x:auto"><table style="border-collapse:collapse;width:100%;min-width:640px;font-size:14px">'
            f'<thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table></div></div>')


# ------------------------------------------------------------------ generated guides and prompt evaluation
SCORE_LABELS = {"numbers_real": "Numbers match the facts", "products_covered": "Products covered", "complaints_covered": "Complaints covered",
                "length_ok": "Length", "structure_ok": "Structure", "names_ok": "No unknown product names"}


def show_guide(cat_name):
    article = ARTICLES.get(cat_name)
    if not article:
        return "", ("No article found for this category. Upload `category_articles.md` (and `category_facts.json`) from your notebook "
                    "to the Space's Files tab, next to `app.py`.")
    chips = ""
    if EVAL is not None and BEST_VARIANT is not None:
        row = EVAL[(EVAL["category"] == cat_name) & (EVAL["variant"] == BEST_VARIANT)]
        if len(row):
            r = row.iloc[0]
            chips = "".join(
                f'<span style="display:inline-block;margin:0 8px 8px 0;padding:6px 12px;border-radius:10px;border:1px solid {RULE};font-size:14px;color:{TEXT}">'
                f'{SCORE_LABELS[c]}: <b>{float(r[c]):.2f}</b></span>' for c in SCORE_COLS)
            chips = (f'<div style="margin-bottom:4px;font-size:14px;color:{MUTED}">Scores from your prompt evaluation, prompt variant {esc(BEST_VARIANT)} '
                     f'(1.00 is best):</div>{chips}')
    return chips, article


def eval_tables():
    if EVAL is None:
        return None, None
    cols = SCORE_COLS + [c for c in ["overall", "words"] if c in EVAL.columns]
    summary = EVAL.groupby("variant")[cols].mean().round(2).reset_index()
    detail_cols = ["variant", "category"] + cols + (["unsupported"] if "unsupported" in EVAL.columns else [])
    detail = EVAL[detail_cols].copy()
    detail[cols] = detail[cols].round(2)
    return summary, detail


def sources_note():
    def flag(ok, name):
        return f"{name}: {'found' if ok else 'not found'}"
    return ("*Data sources. " + "; ".join([flag(bool(NB_FACTS), "category_facts.json"), flag(bool(ARTICLES), "category_articles.md"),
                                           flag(EVAL is not None, "prompt_evaluation.csv")]) +
            (f". {PRODUCTS_UPDATED} products use your notebook's numbers." if PRODUCTS_UPDATED else ". Built-in numbers are used for the products.") + "*")


# ------------------------------------------------------------------ event handlers
def on_category(cat_name):
    rows = table_rows(cat_name)
    summary, charts, quotes, name = render_product(cat_name, rows[0][0])
    return rows, summary, charts, quotes, name, "", ""


def on_select(cat_name, evt: gr.SelectData):
    rows = table_rows(cat_name)
    name = None
    row_value = getattr(evt, "row_value", None)
    if row_value:
        name = row_value[0]
    if not name:
        idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        name = rows[idx][0]
    summary, charts, quotes, name = render_product(cat_name, name)
    return summary, charts, quotes, name, "", ""


# ------------------------------------------------------------------ UI
def build_ui():
    with gr.Blocks(title="Review Explorer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Review Explorer\nPick a category, then **click a product** in the table to read its review summary. "
                    f"Based on {DATA['total']:,} Amazon device reviews, grouped into {len(CAT_NAMES)} categories.")
        with gr.Tabs():
            with gr.Tab("Explore products"):
                selected = gr.State("")
                with gr.Row():
                    with gr.Column(scale=2, min_width=320):
                        category = gr.Dropdown(choices=CAT_NAMES, value=CAT_NAMES[0], label="Category")
                        table = gr.Dataframe(value=table_rows(CAT_NAMES[0]), headers=["Product", "Reviews", "Avg rating", "Negative"],
                                             datatype=["str", "number", "number", "str"], interactive=False, wrap=True,
                                             label="Products (click a row)")
                    with gr.Column(scale=3, min_width=360):
                        summary = gr.Markdown()
                        charts = gr.HTML()
                        quotes = gr.HTML()
                with gr.Accordion("Write the summary with AI (optional)", open=False):
                    gr.Markdown("A small language model rewrites the facts as prose. The numbers are checked against the fact sheet afterwards. "
                                "The first run loads the model and can take a minute.")
                    ai_btn = gr.Button("Write summary with AI")
                    ai_out = gr.Markdown()
                with gr.Accordion("Check the sentiment model on this product's reviews (optional)", open=False):
                    sent_btn = gr.Button("Run the sentiment model")
                    sent_out = gr.HTML()
                outputs = [table, summary, charts, quotes, selected, ai_out, sent_out]
                category.change(on_category, category, outputs)
                demo.load(on_category, category, outputs)
                table.select(on_select, category, [summary, charts, quotes, selected, ai_out, sent_out])
                ai_btn.click(ai_summary, [category, selected], ai_out)
                sent_btn.click(sentiment_check, [category, selected], sent_out)
            with gr.Tab("Compare categories"):
                gr.HTML(compare_html())
            with gr.Tab("Generated guides"):
                gr.Markdown("The buying guide the language model wrote for each category in the Task 3 notebook, with its evaluation scores.")
                g_cat = gr.Dropdown(choices=CAT_NAMES, value=CAT_NAMES[0], label="Category")
                g_scores = gr.HTML()
                g_article = gr.Markdown()
                g_cat.change(show_guide, g_cat, [g_scores, g_article])
                demo.load(show_guide, g_cat, [g_scores, g_article])
            with gr.Tab("Prompt evaluation"):
                summary_df, detail_df = eval_tables()
                if summary_df is None:
                    gr.Markdown("Upload `prompt_evaluation.csv` from your notebook to the Space's Files tab to see the prompt comparison here.")
                else:
                    gr.Markdown(f"Three prompt variants were scored on every category. Best average score: **variant {BEST_VARIANT}**. "
                                "Scores run from 0 to 1, higher is better. A high overall score with a low *numbers match* score means invented numbers.")
                    gr.Dataframe(value=summary_df, interactive=False, label="Average by prompt variant")
                    gr.Dataframe(value=detail_df, interactive=False, wrap=True, label="Every article")
        gr.Markdown(sources_note())
        gr.Markdown("*Summaries come from keyword themes over 1-2 star (complaints) and 4-5 star (praise) reviews. "
                    "Product names were verified by hand because the dataset's name column is unreliable.*")
    return demo


if __name__ == "__main__":
    build_ui().launch()
