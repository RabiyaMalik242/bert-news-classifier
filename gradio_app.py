"""
News Topic Classifier — BERT Fine-tuned on AG News
Gradio Deployment 
Author: Rabiya Malik
"""

import gradio as gr
import torch
import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    pipeline
)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
MODEL_DIR  = './bert_news_classifier_final'
MAX_LENGTH = 128
DEVICE     = 0 if torch.cuda.is_available() else -1

LABEL_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']

LABEL_META = {
    'World':    {'icon': '🌍', 'color': '#3498db', 'desc': 'International news & geopolitics'},
    'Sports':   {'icon': '🏅', 'color': '#e74c3c', 'desc': 'Sports events & athlete news'},
    'Business': {'icon': '💼', 'color': '#2ecc71', 'desc': 'Finance, markets & corporate news'},
    'Sci/Tech': {'icon': '💻', 'color': '#f39c12', 'desc': 'Science, technology & innovation'},
}

EXAMPLE_HEADLINES = [
    ["Pakistan air force conducts joint exercise with Turkey"],
    ["Real Madrid wins Champions League final against Manchester City"],
    ["Tesla reports record quarterly deliveries of 500,000 vehicles"],
    ["NASA's James Webb Telescope captures image of distant galaxy cluster"],
    ["United Nations calls emergency meeting over Middle East tensions"],
    ["Apple unveils M4 chip with enhanced AI processing capabilities"],
    ["Federal Reserve holds interest rates steady amid inflation concerns"],
    ["Olympic champion breaks 100m world record at Paris Athletics"],
    ["Google DeepMind solves decades-old protein folding challenge"],
    ["China and US sign new trade agreement after months of negotiations"],
]

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print(f'Loading model from {MODEL_DIR}...')
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model     = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    print('Model loaded successfully.')
except Exception as e:
    print(f'ERROR: Could not load model — {e}')
    print('Make sure you have run the notebook and saved the model first.')
    raise

# HuggingFace pipeline for clean inference
classifier = pipeline(
    'text-classification',
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    top_k=None
)

# ─────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────
def classify_headline(headline: str):
    """
    Classify a news headline into one of four AG News categories.
    Returns:
      - label_probs (dict)  : for Gradio Label component
      - result_html (str)   : styled HTML result card
    """
    if not headline or not headline.strip():
        return (
            {label: 0.0 for label in LABEL_NAMES},
            "<p style='color:#888;'>Please enter a headline above.</p>"
        )

    # Run inference
    output = classifier(headline.strip(), truncation=True, max_length=MAX_LENGTH)

    print("\nOUTPUT TYPE:", type(output))
    print("OUTPUT:", output)

    results = output[0]

    print("RESULTS TYPE:", type(results))
    print("RESULTS:", results)

    probs = {r['label']: float(r['score']) for r in results}

    # Top prediction
    top_label = max(probs, key=probs.get)
    top_score = probs[top_label]
    meta      = LABEL_META[top_label]

    # Confidence tier
    if top_score >= 0.95:
        confidence_text = 'Very High Confidence'
        conf_color = '#2ecc71'
    elif top_score >= 0.80:
        confidence_text = 'High Confidence'
        conf_color = '#3498db'
    elif top_score >= 0.60:
        confidence_text = 'Moderate Confidence'
        conf_color = '#f39c12'
    else:
        confidence_text = 'Low Confidence'
        conf_color = '#e74c3c'

    # Build HTML result card
    html = f"""
    <div style="
        background: linear-gradient(135deg, {meta['color']}15, {meta['color']}05);
        border-left: 5px solid {meta['color']};
        border-radius: 10px;
        padding: 20px 24px;
        margin: 8px 0;
        font-family: 'Segoe UI', sans-serif;
    ">
        <div style="font-size: 2.2rem; margin-bottom: 6px;">{meta['icon']}</div>
        <div style="font-size: 1.5rem; font-weight: 700; color: {meta['color']};">
            {top_label}
        </div>
        <div style="color: #666; font-size: 0.9rem; margin: 4px 0 12px 0;">
            {meta['desc']}
        </div>
        <div style="
            display: inline-block;
            background: {conf_color}20;
            color: {conf_color};
            border: 1px solid {conf_color};
            border-radius: 20px;
            padding: 3px 14px;
            font-size: 0.82rem;
            font-weight: 600;
        ">
            {confidence_text} — {top_score*100:.1f}%
        </div>

        <div style="margin-top: 16px;">
            <div style="font-size: 0.8rem; color: #888; margin-bottom: 8px;">
                ALL CLASS PROBABILITIES
            </div>
    """

    # Probability bars for each class
    for label in LABEL_NAMES:
        score = probs[label]
        lmeta = LABEL_META[label]
        bar_width = int(score * 100)
        is_top = label == top_label
        html += f"""
            <div style="margin: 5px 0;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="width:70px; font-size:0.82rem;
                                 font-weight:{'700' if is_top else '400'};
                                 color:{'#333' if is_top else '#888'};">
                        {lmeta['icon']} {label}
                    </span>
                    <div style="flex:1; background:#f0f0f0; border-radius:4px; height:10px;">
                        <div style="
                            width:{bar_width}%;
                            background:{lmeta['color']};
                            height:10px;
                            border-radius:4px;
                            opacity:{'1.0' if is_top else '0.45'};
                        "></div>
                    </div>
                    <span style="width:45px; text-align:right; font-size:0.82rem;
                                 font-weight:{'700' if is_top else '400'};
                                 color:{'#333' if is_top else '#999'};">
                        {score*100:.1f}%
                    </span>
                </div>
            </div>
        """

    html += """
        </div>
        <div style="margin-top:14px; font-size:0.75rem; color:#aaa;">
            Model: bert-base-uncased fine-tuned on AG News •
        </div>
    </div>
    """

    return probs, html


# ─────────────────────────────────────────────
# GRADIO INTERFACE
# ─────────────────────────────────────────────
custom_css = """
.gradio-container {
    max-width: 860px !important;
    margin: auto !important;
    font-family: 'Segoe UI', sans-serif !important;
}
.title-area {
    text-align: center;
    padding: 10px 0 4px 0;
}
footer { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue='blue',
        secondary_hue='slate',
        neutral_hue='slate'
    ),
    css=custom_css,
    title='News Topic Classifier — BERT'
) as demo:

    # ── Header ──────────────────────────────
    gr.HTML("""
    <div class="title-area">
        <h1 style="font-size:1.9rem; margin-bottom:4px;">
            🗞️ News Topic Classifier
        </h1>
        <p style="color:#666; font-size:0.95rem; margin:0;">
            Fine-tuned <code>bert-base-uncased</code> on AG News (120,000 articles)
        </p>
        <div style="display:flex; justify-content:center; gap:16px; margin-top:10px; flex-wrap:wrap;">
            <span style="background:#3498db15; color:#3498db; border:1px solid #3498db;
                         padding:3px 14px; border-radius:20px; font-size:0.82rem;">
                🌍 World
            </span>
            <span style="background:#e74c3c15; color:#e74c3c; border:1px solid #e74c3c;
                         padding:3px 14px; border-radius:20px; font-size:0.82rem;">
                🏅 Sports
            </span>
            <span style="background:#2ecc7115; color:#2ecc71; border:1px solid #2ecc71;
                         padding:3px 14px; border-radius:20px; font-size:0.82rem;">
                💼 Business
            </span>
            <span style="background:#f39c1215; color:#f39c12; border:1px solid #f39c12;
                         padding:3px 14px; border-radius:20px; font-size:0.82rem;">
                💻 Sci/Tech
            </span>
        </div>
    </div>
    """)

    gr.HTML("<hr style='border-color:#e0e0e0; margin:12px 0;'>")

    # ── Input ───────────────────────────────
    with gr.Row():
        with gr.Column(scale=4):
            headline_input = gr.Textbox(
                label='News Headline',
                placeholder='Enter a news headline to classify...',
                lines=2,
                max_lines=4,
                show_label=True
            )
        with gr.Column(scale=1, min_width=120):
            classify_btn = gr.Button(
                '🔍 Classify',
                variant='primary',
                size='lg'
            )

    # ── Outputs ─────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            label_output = gr.Label(
                label='Class Probabilities',
                num_top_classes=4
            )
        with gr.Column(scale=1):
            result_html = gr.HTML(
                label='Result',
                value="<p style='color:#aaa; padding:20px;'>Result will appear here.</p>"
            )

    # ── Examples ────────────────────────────
    gr.HTML("<hr style='border-color:#e0e0e0; margin:12px 0;'>")
    gr.HTML("<p style='color:#888; font-size:0.85rem; margin:0 0 8px 0;'>📋 Try these example headlines:</p>")

    gr.Examples(
        examples=EXAMPLE_HEADLINES,
        inputs=headline_input,
        outputs=[label_output, result_html],
        fn=classify_headline,
        cache_examples=False,
        examples_per_page=5
    )

    # ── Model Info ──────────────────────────
    with gr.Accordion('ℹ️ Model Information', open=False):
        gr.Markdown(f"""
        | Property | Detail |
        |---|---|
        | Base Model | `bert-base-uncased` (110M parameters) |
        | Dataset | AG News (120,000 training / 7,600 test) |
        | Training Subset | 8,000 samples (stratified, 2,000 per class) |
        | Fine-tuning | 3 epochs, lr=2e-5, batch_size=16 |
        | Expected Accuracy | ~93–95% on AG News test set |
        | Tokenizer | BERT WordPiece, max_length=128 |
        | Device | {'GPU' if torch.cuda.is_available() else 'CPU'} |
        | Framework | PyTorch + HuggingFace Transformers |

        **AG News Categories:**
        - 🌍 **World** — International news, politics, geopolitics
        - 🏅 **Sports** — All sports, athletes, tournaments
        - 💼 **Business** — Finance, markets, corporate, economy
        - 💻 **Sci/Tech** — Science, technology, AI, space exploration
        """)

    # ── Wire up events ──────────────────────
    classify_btn.click(
        fn=classify_headline,
        inputs=headline_input,
        outputs=[label_output, result_html]
    )

    headline_input.submit(
        fn=classify_headline,
        inputs=headline_input,
        outputs=[label_output, result_html]
    )


# ─────────────────────────────────────────────
# LAUNCH
# ─────────────────────────────────────────────
if __name__ == '__main__':
    demo.launch(
        server_name='127.0.0.1',
        share=False,
        show_error=True
    )
