import streamlit as st
import pickle
import numpy as np
import pandas as pd
import nltk
import io
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    TF_IMPORT_ERROR = None
except Exception as exc:
    tf = None
    pad_sequences = None
    TF_IMPORT_ERROR = exc

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
ARTIFACTS_DIR = BASE_DIR / 'artifacts'
MAX_WORDS = 10000
MAX_LEN   = 50
LABELS    = ['Irrelevant', 'Negative', 'Neutral', 'Positive']
ASSET_FILES = [
    MODELS_DIR / 'sentiment_model.h5',
    MODELS_DIR / 'sentiment_model.keras',
    MODELS_DIR / 'sentiment_weights.weights.h5',
    MODELS_DIR / 'tokenizer.pickle',
]

# ── Design tokens ─────────────────────────────────────────────────────────────
BG       = '#111010'
SURFACE  = '#1A1918'
SURFACE2 = '#221F1E'
BORDER   = '#2E2B29'
AMBER    = '#C8973A'
AMBER_DIM= '#8A6525'
CREAM    = '#E8E0D0'
MUTED    = '#6B6560'
RED      = '#B85C4A'
TEAL     = '#4A8A7E'
SLATE    = '#8A9BA8'

CLASS_COLORS = {
    'Irrelevant': SLATE,
    'Negative':   RED,
    'Neutral':    TEAL,
    'Positive':   '#7AAF6E',
}

CHART_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='IBM Plex Mono, monospace', color=MUTED, size=11),
    xaxis=dict(gridcolor='#221F1E', linecolor=BORDER, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor='#221F1E', linecolor=BORDER, tickfont=dict(color=MUTED)),
    margin=dict(l=10, r=10, t=44, b=10),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=BORDER,
                borderwidth=1, font=dict(color=MUTED, size=11)),
    title_font=dict(family='Barlow Condensed, sans-serif', color=MUTED, size=11,
                    ),
)

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Serif:ital,wght@0,300;0,400;0,600;1,300&family=Barlow+Condensed:wght@300;400;600;700&display=swap');

:root {{
    --bg:      {BG};
    --surf:    {SURFACE};
    --surf2:   {SURFACE2};
    --border:  {BORDER};
    --amber:   {AMBER};
    --amber-d: {AMBER_DIM};
    --cream:   {CREAM};
    --muted:   {MUTED};
    --red:     {RED};
    --teal:    {TEAL};
}}

html, body, [data-testid="stAppViewContainer"] {{
    background: var(--bg) !important;
    color: var(--cream);
}}
* {{ font-family: 'IBM Plex Mono', monospace; box-sizing: border-box; }}

/* masthead */
.masthead {{
    border-bottom: 1px solid var(--border);
    padding: 2rem 0 1.4rem;
    margin-bottom: 0;
}}
.masthead-label {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--amber);
    margin-bottom: 0.4rem;
}}
.masthead h1 {{
    font-family: 'IBM Plex Serif', serif;
    font-size: 2rem;
    font-weight: 300;
    color: var(--cream);
    letter-spacing: -.01em;
    margin: 0;
    line-height: 1.2;
}}
.masthead h1 em {{ font-style: italic; color: var(--amber); }}
.masthead-sub {{
    font-size: 0.73rem;
    color: var(--muted);
    margin: 0.5rem 0 0;
    letter-spacing: .04em;
}}

/* section rule */
.rule-heading {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1rem;
}}
.rule-label {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--muted);
    white-space: nowrap;
}}
.rule-line {{
    flex: 1;
    height: 1px;
    background: var(--border);
}}

/* stat strip */
.stat-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 1px;
    border: 1px solid var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin: 1.2rem 0;
}}
.stat-cell {{
    background: var(--surf);
    padding: 1rem 1rem 0.9rem;
    border-bottom: 2px solid transparent;
    transition: border-color .2s;
}}
.stat-cell:hover {{ border-bottom-color: var(--amber); }}
.stat-val {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--cream);
    line-height: 1;
}}
.stat-lbl {{
    font-size: 0.62rem;
    color: var(--muted);
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}}

/* card */
.card {{
    background: var(--surf);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 1.2rem 1.4rem;
    margin: 0.5rem 0;
}}

/* pipeline step */
.pipe-item {{
    display: flex;
    align-items: flex-start;
    gap: 1.2rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid var(--border);
}}
.pipe-item:last-child {{ border-bottom: none; }}
.pipe-num {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--amber-d);
    line-height: 1;
    min-width: 26px;
}}
.pipe-title {{ font-size: 0.8rem; font-weight: 500; color: var(--cream); }}
.pipe-desc  {{ font-size: 0.74rem; color: var(--muted); margin-top: 0.15rem; line-height: 1.5; }}

/* result banner */
.result-banner {{
    border: 1px solid var(--border);
    border-left-width: 3px;
    background: var(--surf);
    border-radius: 3px;
    padding: 0.9rem 1.4rem;
    margin: 0.8rem 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.5rem;
}}
.result-label {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: .06em;
}}
.result-conf {{ font-size: 0.76rem; color: var(--muted); }}

/* asset table */
.asset-table {{ width: 100%; border-collapse: collapse; font-size: 0.76rem; }}
.asset-table th {{
    text-align: left;
    font-size: 0.62rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    border-bottom: 1px solid var(--amber-d);
    padding: 0.4rem 0.6rem;
}}
.asset-table td {{
    padding: 0.5rem 0.6rem;
    border-bottom: 1px solid var(--border);
    color: var(--cream);
    vertical-align: middle;
}}
.asset-table tr:last-child td {{ border-bottom: none; }}
.ok  {{ color: #7AAF6E; }}
.err {{ color: var(--red); }}

/* sample note */
.sample-note {{
    font-size: 0.66rem;
    color: var(--muted);
    letter-spacing: .06em;
    text-align: right;
    margin-top: 0.3rem;
    border-top: 1px solid var(--border);
    padding-top: 0.35rem;
}}

/* class grid */
.class-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1px;
    border: 1px solid var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin: 0.8rem 0;
}}
.class-cell {{
    background: var(--surf);
    padding: 0.8rem 1rem;
    border-top: 2px solid;
    font-size: 0.8rem;
    font-weight: 500;
}}

/* developer card */
.dev-wrap {{
    display: grid;
    grid-template-columns: 160px 1fr;
    gap: 0;
    border: 1px solid var(--border);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 1.5rem;
}}
.dev-sidebar {{
    background: var(--surf2);
    padding: 2rem 1.4rem;
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.8rem;
}}
.dev-avatar {{
    width: 76px;
    height: 76px;
    border-radius: 2px;
    object-fit: cover;
    filter: grayscale(20%) sepia(20%);
    border: 1px solid var(--border);
}}
.dev-monogram {{
    font-size: 0.6rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--muted);
    text-align: center;
}}
.dev-body {{
    background: var(--surf);
    padding: 1.8rem 2rem;
}}
.dev-name {{
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.3rem;
    font-weight: 300;
    color: var(--cream);
    margin: 0 0 0.25rem;
}}
.dev-title {{
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}}
.dev-tagline {{
    font-family: 'IBM Plex Serif', serif;
    font-style: italic;
    font-size: 0.85rem;
    color: var(--amber-d);
    margin-bottom: 1.2rem;
    border-left: 2px solid var(--amber-d);
    padding-left: 0.8rem;
}}
.dev-links {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
}}
.dev-link {{
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: .08em;
    text-transform: uppercase;
    padding: 0.35rem 0.85rem;
    border: 1px solid var(--border);
    border-radius: 2px;
    color: var(--muted);
    text-decoration: none;
    transition: color .15s, border-color .15s;
    background: transparent;
}}
.dev-link:hover {{ color: var(--amber); border-color: var(--amber-d); }}

/* streamlit overrides */
[data-testid="metric-container"] {{
    background: var(--surf) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    padding: 0.85rem 1rem !important;
}}
[data-baseweb="tab-list"] {{
    background: var(--surf) !important;
    border-radius: 3px !important;
    padding: 6px 8px !important;
    gap: 10px !important;
    border: 1px solid var(--border) !important;
    overflow-x: auto !important;
}}
[data-baseweb="tab"] {{
    border-radius: 2px !important;
    font-size: 0.76rem !important;
    letter-spacing: .05em !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--muted) !important;
    padding: 0.45rem 0.9rem !important;
    min-height: auto !important;
}}
[aria-selected="true"][data-baseweb="tab"] {{
    background: var(--amber) !important;
    color: {BG} !important;
    font-weight: 600 !important;
}}
textarea, input[type="text"] {{
    background: var(--surf) !important;
    border: 1px solid var(--border) !important;
    color: var(--cream) !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
}}
.stButton > button {{
    background: var(--amber) !important;
    color: {BG} !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: .06em !important;
    font-size: 0.8rem !important;
}}
.stButton > button:hover {{ opacity: 0.85 !important; }}
[data-testid="stDataFrame"] thead tr th {{
    background: var(--surf2) !important;
    color: var(--muted) !important;
    font-size: 0.68rem !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    font-family: 'IBM Plex Mono', monospace !important;
}}
[data-testid="stDataFrame"] td {{
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    color: var(--cream) !important;
}}
div[data-testid="stExpander"] {{
    background: var(--surf) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
}}
div[data-testid="stExpander"] summary {{
    font-size: 0.78rem !important;
    color: var(--muted) !important;
}}
[data-testid="stFileUploaderDropzone"] {{
    background: var(--surf) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 3px !important;
}}
</style>
"""


# ── UI helpers ────────────────────────────────────────────────────────────────
def rule(label):
    st.markdown(
        f'<div class="rule-heading">'
        f'<span class="rule-label">{label}</span>'
        f'<div class="rule-line"></div></div>',
        unsafe_allow_html=True,
    )


def stat_row(items):
    cells = ''.join(
        f'<div class="stat-cell">'
        f'<div class="stat-val">{v}</div>'
        f'<div class="stat-lbl">{l}</div>'
        f'</div>'
        for v, l in items
    )
    st.markdown(f'<div class="stat-row">{cells}</div>', unsafe_allow_html=True)


def sample_note(msg='Illustrative sample data — replace with real files to show actual numbers'):
    st.markdown(f'<div class="sample-note">{msg}</div>', unsafe_allow_html=True)


# ── NLTK ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def ensure_nltk():
    for path, name in {
        'tokenizers/punkt':     'punkt',
        'tokenizers/punkt_tab': 'punkt_tab',
        'corpora/wordnet':      'wordnet',
        'corpora/stopwords':    'stopwords',
    }.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


@st.cache_resource
def get_text_tools():
    return set(stopwords.words('english')), WordNetLemmatizer()


def clean_text(text):
    if not isinstance(text, str):
        return ''
    stop_words, lem = get_text_tools()
    tokens = word_tokenize(text.lower())
    return ' '.join(lem.lemmatize(w) for w in tokens
                    if w.isalpha() and w not in stop_words)


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    m = tf.keras.Sequential([
        tf.keras.layers.Embedding(MAX_WORDS, 64, input_shape=(MAX_LEN,)),
        tf.keras.layers.SpatialDropout1D(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.2)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(4,  activation='softmax'),
    ])
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return m


@st.cache_resource
def load_assets():
    if tf is None or pad_sequences is None:
        raise RuntimeError(
            f'TensorFlow is not available in this environment. Original import error: {TF_IMPORT_ERROR}'
        )
    with open(MODELS_DIR / 'tokenizer.pickle', 'rb') as fh:
        tok = pickle.load(fh)
    model = build_model()
    model.load_weights(MODELS_DIR / 'sentiment_weights.weights.h5')
    return model, tok


def model_summary_str(model):
    buf = io.StringIO()
    model.summary(print_fn=lambda l: buf.write(l + '\n'))
    return buf.getvalue()


def asset_df():
    rows = []
    for p in ASSET_FILES:
        ok  = p.exists()
        rows.append({
            'File':     str(p.relative_to(BASE_DIR)).replace('\\', '/'),
            'Found':    'yes' if ok else 'no',
            'Size MB':  round(p.stat().st_size / 1_048_576, 3) if ok else '—',
            'Modified': datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M') if ok else '—',
        })
    return pd.DataFrame(rows)


# ── Training data (real or reproducible sample) ───────────────────────────────
@st.cache_data
def load_history():
    p = ARTIFACTS_DIR / 'training_history.json'
    if p.exists():
        with open(p) as fh:
            return json.load(fh), False
    rng = np.random.default_rng(42)
    n   = 15
    tl  = [max(0.04, 0.98 * (0.72 ** i) + rng.normal(0, .012)) for i in range(n)]
    vl  = [max(0.08, 1.10 * (0.76 ** i) + rng.normal(0, .018)) for i in range(n)]
    ta  = [min(0.99, 1 - 0.55 * (0.72 ** i) + rng.normal(0, .008)) for i in range(n)]
    va  = [min(0.99, 1 - 0.60 * (0.76 ** i) + rng.normal(0, .012)) for i in range(n)]
    return {'loss': tl, 'val_loss': vl, 'accuracy': ta, 'val_accuracy': va}, True


@st.cache_data
def load_metrics():
    p = ARTIFACTS_DIR / 'eval_metrics.json'
    if p.exists():
        with open(p) as fh:
            return json.load(fh), False
    return {
        'train': {'accuracy': 0.9210, 'loss': 0.2140, 'precision': 0.9195, 'recall': 0.9210, 'f1': 0.9199},
        'test':  {'accuracy': 0.8710, 'loss': 0.3580, 'precision': 0.8690, 'recall': 0.8710, 'f1': 0.8696},
        'per_class': {'test': {
            'Irrelevant': {'precision': 0.850, 'recall': 0.820, 'f1': 0.835, 'support': 320},
            'Negative':   {'precision': 0.910, 'recall': 0.890, 'f1': 0.900, 'support': 480},
            'Neutral':    {'precision': 0.830, 'recall': 0.860, 'f1': 0.845, 'support': 540},
            'Positive':   {'precision': 0.880, 'recall': 0.900, 'f1': 0.890, 'support': 460},
        }},
    }, True


# ── Chart builders ─────────────────────────────────────────────────────────────
def chart_curves(history):
    acc_key = 'accuracy' if 'accuracy' in history else 'acc'
    val_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
    epochs  = list(range(1, len(history['loss']) + 1))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('LOSS', 'ACCURACY'),
        horizontal_spacing=0.12,
    )
    # --- loss ---
    fig.add_trace(go.Scatter(
        x=epochs, y=history['loss'], name='Train',
        line=dict(color=AMBER, width=2),
        mode='lines+markers', marker=dict(size=4, color=AMBER),
    ), row=1, col=1)
    if 'val_loss' in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history['val_loss'], name='Validation',
            line=dict(color=RED, width=2, dash='dot'),
            mode='lines+markers', marker=dict(size=4, color=RED),
        ), row=1, col=1)
    # --- accuracy ---
    fig.add_trace(go.Scatter(
        x=epochs, y=history[acc_key], name='Train',
        line=dict(color=AMBER, width=2),
        mode='lines+markers', marker=dict(size=4, color=AMBER),
        showlegend=False,
    ), row=1, col=2)
    if val_key in history:
        fig.add_trace(go.Scatter(
            x=epochs, y=history[val_key], name='Validation',
            line=dict(color=RED, width=2, dash='dot'),
            mode='lines+markers', marker=dict(size=4, color=RED),
            showlegend=False,
        ), row=1, col=2)

    fig.update_layout(height=340, **CHART_BASE, title_text=None)
    fig.update_xaxes(title_text='Epoch', gridcolor=SURFACE2, linecolor=BORDER,
                     tickfont=dict(color=MUTED), title_font=dict(color=MUTED))
    fig.update_yaxes(gridcolor=SURFACE2, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(title_text='Loss',     title_font=dict(color=MUTED), row=1, col=1)
    fig.update_yaxes(title_text='Accuracy', title_font=dict(color=MUTED), row=1, col=2)
    for ann in fig.layout.annotations:
        ann.font = dict(family='Barlow Condensed, sans-serif', size=10, color=MUTED)
    return fig


def chart_radar(metrics):
    keys = ['accuracy', 'precision', 'recall', 'f1']
    tr   = [metrics['train'].get(k, 0) for k in keys]
    te   = [metrics['test'].get(k, 0)  for k in keys]
    fig  = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=tr + [tr[0]], theta=keys + [keys[0]],
        fill='toself', name='Train',
        line=dict(color=AMBER, width=2),
        fillcolor='rgba(200,151,58,.15)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=te + [te[0]], theta=keys + [keys[0]],
        fill='toself', name='Test',
        line=dict(color=RED, width=2, dash='dot'),
        fillcolor='rgba(184,92,74,.15)',
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0.7, 1.0],
                            gridcolor=BORDER, color=MUTED,
                            tickfont=dict(size=9, color=MUTED)),
            angularaxis=dict(gridcolor=BORDER, color=MUTED,
                             tickfont=dict(family='Barlow Condensed', size=12, color=CREAM)),
            bgcolor='rgba(0,0,0,0)',
        ),
        height=320,
        title_text='TRAIN VS TEST — RADAR',
        **CHART_BASE,
    )
    return fig


def chart_per_class(metrics):
    pc   = metrics.get('per_class', {}).get('test', {})
    prec = [pc.get(l, {}).get('precision', 0) for l in LABELS]
    rec  = [pc.get(l, {}).get('recall',    0) for l in LABELS]
    f1   = [pc.get(l, {}).get('f1',        0) for l in LABELS]

    fig = go.Figure()
    for vals, name, clr in [
        (prec, 'Precision', AMBER),
        (rec,  'Recall',    TEAL),
        (f1,   'F1',        RED),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=LABELS, y=vals,
            marker_color=clr,
            marker_line_color='rgba(0,0,0,0)',
            text=[f'{v:.3f}' for v in vals],
            textposition='outside',
            textfont=dict(color=MUTED, size=10),
        ))
    fig.update_layout(
        barmode='group', height=310,
        title_text='PER-CLASS METRICS (TEST SET)',
        yaxis_range=[0, 1.12],
        **CHART_BASE,
    )
    return fig


def chart_params(model):
    names  = [l.name for l in model.layers]
    params = [l.count_params() for l in model.layers]
    fig = go.Figure(go.Bar(
        x=params, y=names, orientation='h',
        marker=dict(
            color=params,
            colorscale=[[0, SURFACE2], [0.35, AMBER_DIM], [1, AMBER]],
            showscale=False,
        ),
        text=[f'{p:,}' for p in params],
        textposition='outside',
        textfont=dict(color=MUTED, size=10),
    ))
    fig.update_layout(
        height=270,
        title_text='PARAMETERS PER LAYER',
        xaxis_title='Count',
        **CHART_BASE,
    )
    return fig


def chart_prob_bar(preds):
    clrs = [CLASS_COLORS[l] for l in LABELS]
    fig = go.Figure(go.Bar(
        x=LABELS, y=preds,
        marker_color=clrs,
        marker_line_color='rgba(0,0,0,0)',
        text=[f'{p*100:.1f}%' for p in preds],
        textposition='outside',
        textfont=dict(color=MUTED, size=11),
    ))
    fig.update_layout(
        height=270,
        title_text='CLASS PROBABILITY DISTRIBUTION',
        yaxis_range=[0, 1.18],
        yaxis_tickformat='.0%',
        **CHART_BASE,
    )
    return fig


# ── Dataset helpers ───────────────────────────────────────────────────────────
@st.cache_data
def load_training_datasets():
    """Load the exact datasets used in training/validation."""
    columns = ['id', 'entity', 'sentiment', 'text']
    files = {
        'Training': DATA_DIR / 'twitter_training.csv',
        'Validation': DATA_DIR / 'twitter_validation.csv',
    }

    loaded = {}
    for split_name, file_path in files.items():
        if not file_path.exists():
            loaded[split_name] = None
            continue

        df = pd.read_csv(
            file_path,
            header=None,
            names=columns,
            encoding='utf-8',
            on_bad_lines='skip'
        )
        loaded[split_name] = df

    return loaded


@st.cache_data
def read_dataset(fp):
    p   = Path(fp)
    ext = p.suffix.lower()
    if ext == '.csv':      return pd.read_csv(p)
    if ext == '.json':     return pd.read_json(p)
    if ext == '.parquet':  return pd.read_parquet(p)
    if ext == '.xlsx':     return pd.read_excel(p)
    raise ValueError(f'Unsupported: {ext}')


# ── Page renders ──────────────────────────────────────────────────────────────
def render_overview():
    st.markdown("""
    <div class="card" style="margin-top:1rem">
      <span style="font-family:'IBM Plex Serif',serif;font-style:italic;
                   color:#8A6525;font-size:.8rem;display:block;margin-bottom:.5rem">
        What this studio does
      </span>
      <p style="margin:0;font-size:.8rem;color:#6B6560;line-height:1.75">
        A complete sentiment analysis workbench built on a Bidirectional LSTM trained on tweet data.
        Every aspect of the model — training dynamics, evaluation performance, architecture, and data
        distribution — is surfaced here with no abstraction layers between you and the numbers.
      </p>
    </div>
    """, unsafe_allow_html=True)

    stat_row([
        (len(LABELS),      'Sentiment Classes'),
        (f'{MAX_WORDS:,}', 'Vocabulary Size'),
        (MAX_LEN,          'Max Sequence Length'),
        ('BiLSTM',         'Architecture'),
        ('Adam',           'Optimizer'),
        ('Cat. CE',        'Loss Function'),
    ])

    rule('Sentiment Classes')
    cells = ''.join(
        f'<div class="class-cell" style="border-top-color:{CLASS_COLORS[l]};color:{CLASS_COLORS[l]}">{l}</div>'
        for l in LABELS
    )
    st.markdown(f'<div class="class-grid">{cells}</div>', unsafe_allow_html=True)

    rule('Navigate')
    nav = [
        ('Training Insights', 'Loss and accuracy curves, epoch log, best-epoch summary'),
        ('Evaluation',        'Train vs test breakdown, radar comparison, per-class F1/precision/recall'),
        ('Model Info',        'Architecture, layer table, preprocessing pipeline, asset health'),
        ('Dataset Info',      'Inspect training and validation datasets used for the model'),
        ('Predictor',         'Live inference on any input text'),
    ]
    for title, desc in nav:
        st.markdown(
            f'<div class="pipe-item">'
            f'<div class="pipe-num">—</div>'
            f'<div><div class="pipe-title">{title}</div>'
            f'<div class="pipe-desc">{desc}</div></div></div>',
            unsafe_allow_html=True,
        )


def render_training():
    history, is_sample = load_history()
    acc_key = 'accuracy' if 'accuracy' in history else 'acc'
    val_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'

    n_epochs      = len(history['loss'])
    best_loss_ep  = int(np.argmin(history.get('val_loss', history['loss']))) + 1
    best_acc_ep   = int(np.argmax(history.get(val_key, history[acc_key]))) + 1
    final_vl      = history.get('val_loss', history['loss'])[-1]
    final_va      = history.get(val_key, history[acc_key])[-1]

    rule('Summary')
    stat_row([
        (n_epochs,            'Total Epochs'),
        (best_loss_ep,        'Best Loss Epoch'),
        (best_acc_ep,         'Best Acc Epoch'),
        (f'{final_vl:.4f}',   'Final Val Loss'),
        (f'{final_va:.4f}',   'Final Val Acc'),
    ])

    rule('Training Curves')
    st.plotly_chart(chart_curves(history), use_container_width=True)
    if is_sample:
        sample_note()

    rule('Epoch Log')
    rows = {
        'Epoch':      list(range(1, n_epochs + 1)),
        'Train Loss': [round(v, 6) for v in history['loss']],
        'Val Loss':   [round(v, 6) for v in history.get('val_loss', [None] * n_epochs)],
        'Train Acc':  [round(v, 6) for v in history.get(acc_key,    [None] * n_epochs)],
        'Val Acc':    [round(v, 6) for v in history.get(val_key,    [None] * n_epochs)],
    }
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_evaluation(model):
    metrics, is_sample = load_metrics()

    rule('Overall Train vs Test')
    comparison = []
    for k in ['accuracy', 'loss', 'precision', 'recall', 'f1']:
        tr = metrics['train'].get(k)
        te = metrics['test'].get(k)
        if tr is None:
            continue
        delta = (te - tr) if te is not None else None
        comparison.append({
            'Metric':  k.capitalize(),
            'Train':   f'{tr:.4f}',
            'Test':    f'{te:.4f}' if te is not None else '—',
            'Delta':   f'{delta:+.4f}' if delta is not None else '—',
            'Status':  ('Good'     if delta is not None and abs(delta) < 0.04 else
                        'Moderate' if delta is not None and abs(delta) < 0.08 else
                        'High gap'),
        })
    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
    if is_sample:
        sample_note()

    col1, col2 = st.columns(2)
    with col1:
        rule('Radar View')
        st.plotly_chart(chart_radar(metrics), use_container_width=True)
    with col2:
        rule('Per-Class (Test)')
        st.plotly_chart(chart_per_class(metrics), use_container_width=True)

    rule('Per-Class Detail')
    pc = metrics.get('per_class', {}).get('test', {})
    pc_rows = [{
        'Class':     l,
        'Precision': f"{pc.get(l,{}).get('precision', 0):.4f}",
        'Recall':    f"{pc.get(l,{}).get('recall',    0):.4f}",
        'F1':        f"{pc.get(l,{}).get('f1',        0):.4f}",
        'Support':   pc.get(l, {}).get('support', '—'),
    } for l in LABELS]
    st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)

    rule('Parameter Distribution')
    st.plotly_chart(chart_params(model), use_container_width=True)


def render_model_info(model, tokenizer):
    stat_row([
        ('BiLSTM',                         'Architecture'),
        (len(LABELS),                      'Output Classes'),
        (f'{model.count_params():,}',      'Total Parameters'),
        (f'{len(tokenizer.word_index):,}', 'Tokenizer Vocab'),
        (MAX_LEN,                          'Sequence Length'),
        (MAX_WORDS,                        'Vocab Cap'),
    ])

    rule('Architecture Summary')
    with st.expander('model.summary()'):
        st.code(model_summary_str(model), language='text')

    rule('Layer Details')
    layer_rows = []
    for i, layer in enumerate(model.layers, 1):
        out = getattr(layer, 'output_shape', 'n/a')
        layer_rows.append({
            '#':      i,
            'Name':   layer.name,
            'Type':   layer.__class__.__name__,
            'Output': str(out),
            'Params': f'{layer.count_params():,}',
        })
    st.dataframe(pd.DataFrame(layer_rows), use_container_width=True, hide_index=True)

    rule('Configuration')
    st.json({
        'architecture': 'Bidirectional LSTM',
        'max_words': MAX_WORDS,
        'max_sequence_length': MAX_LEN,
        'embedding_dim': 64,
        'lstm_units': 64,
        'dense_hidden': 32,
        'output_classes': len(LABELS),
        'labels': LABELS,
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'tokenizer_vocab_size': int(len(tokenizer.word_index)),
    })

    rule('Preprocessing Pipeline')
    steps = [
        ('Lowercase',      'Convert all characters to lowercase'),
        ('Tokenize',       'Split into word tokens using NLTK punkt tokenizer'),
        ('Filter',         'Remove stopwords and non-alphabetic tokens'),
        ('Lemmatize',      'Reduce each token to its base form (WordNetLemmatizer)'),
        ('Sequence',       'Map token list to integer IDs via the fitted Keras tokenizer'),
        ('Pad / Truncate', f'Enforce fixed length of {MAX_LEN} tokens with post-padding'),
    ]
    for idx, (title, desc) in enumerate(steps, 1):
        st.markdown(
            f'<div class="pipe-item">'
            f'<div class="pipe-num">{idx:02d}</div>'
            f'<div><div class="pipe-title">{title}</div>'
            f'<div class="pipe-desc">{desc}</div></div></div>',
            unsafe_allow_html=True,
        )

    rule('Asset Health')
    df = asset_df()
    header = '<tr><th>File</th><th>Found</th><th>Size MB</th><th>Modified</th></tr>'
    body = ''
    for _, r in df.iterrows():
        cls = 'ok' if r['Found'] == 'yes' else 'err'
        body += (
            f'<tr>'
            f'<td style="color:{CREAM};font-size:.76rem">{r["File"]}</td>'
            f'<td class="{cls}">{r["Found"]}</td>'
            f'<td>{r["Size MB"]}</td>'
            f'<td>{r["Modified"]}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<div class="card" style="padding:0.8rem 1.2rem">'
        f'<table class="asset-table">{header}{body}</table>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_dataset():
    datasets = load_training_datasets()
    missing = [name for name, df in datasets.items() if df is None]
    train_file_path = DATA_DIR / 'twitter_training.csv'
    test_file_path = DATA_DIR / 'twitter_validation.csv'

    if missing:
        st.markdown(
            '<div class="card" style="color:#6B6560;font-size:.8rem">'
            f'Missing required file(s): {", ".join(missing)} dataset. '
            'Expected files are data/twitter_training.csv and data/twitter_validation.csv.'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    rule('Download Datasets')
    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label='Download Training Dataset',
            data=train_file_path.read_bytes(),
            file_name='twitter_training.csv',
            mime='text/csv',
            use_container_width=True,
        )
    with dl_col2:
        st.download_button(
            label='Download Testing Dataset',
            data=test_file_path.read_bytes(),
            file_name='twitter_validation.csv',
            mime='text/csv',
            use_container_width=True,
        )

    df_train = datasets['Training']
    df_valid = datasets['Validation']

    rule('Dataset Size Comparison')
    stat_row([
        (f'{len(df_train):,}', 'Training Rows'),
        (f'{len(df_valid):,}', 'Validation Rows'),
        (len(df_train.columns), 'Columns Per Dataset'),
        (f'{len(df_train) + len(df_valid):,}', 'Total Rows'),
    ])

    rule('Sentiment Value Counts (Both Datasets)')
    train_counts = df_train['sentiment'].astype(str).str.strip().value_counts()
    valid_counts = df_valid['sentiment'].astype(str).str.strip().value_counts()
    label_order = [label for label in LABELS if label in set(train_counts.index) | set(valid_counts.index)]
    if not label_order:
        label_order = sorted(set(train_counts.index) | set(valid_counts.index))

    sentiment_counts = pd.DataFrame({
        'Sentiment': label_order,
        'Training': [int(train_counts.get(lbl, 0)) for lbl in label_order],
        'Validation': [int(valid_counts.get(lbl, 0)) for lbl in label_order],
    })

    fig_counts = go.Figure()
    fig_counts.add_trace(go.Bar(
        name='Training',
        x=sentiment_counts['Sentiment'],
        y=sentiment_counts['Training'],
        marker_color=AMBER,
    ))
    fig_counts.add_trace(go.Bar(
        name='Validation',
        x=sentiment_counts['Sentiment'],
        y=sentiment_counts['Validation'],
        marker_color=TEAL,
    ))
    fig_counts.update_layout(
        barmode='group',
        height=320,
        title_text='SENTIMENT COUNTS: TRAINING VS VALIDATION',
        yaxis_title='Count',
        **CHART_BASE,
    )
    st.plotly_chart(fig_counts, use_container_width=True)
    st.dataframe(sentiment_counts, use_container_width=True, hide_index=True)

    dataset_choice = st.selectbox('Preview dataset', ['Training', 'Validation'])
    df = df_train if dataset_choice == 'Training' else df_valid

    if df is None:
        return

    rule('Dataset Statistics')
    stat_row([
        (f'{len(df):,}',             'Rows'),
        (len(df.columns),            'Columns'),
        (int(df.isna().sum().sum()), 'Missing Values'),
        (int(df.duplicated().sum()), 'Duplicate Rows'),
    ])

    with st.expander('Column types'):
        st.dataframe(pd.DataFrame({
            'Column':   df.columns,
            'Dtype':    df.dtypes.astype(str).values,
            'Non-Null': df.notna().sum().values,
            'Null %':   (df.isna().mean() * 100).round(2).values,
        }), use_container_width=True, hide_index=True)

    label_col = 'sentiment'
    if label_col:
        rule(f'Label Distribution — {label_col}')
        dist = df[label_col].value_counts(dropna=False).reset_index()
        dist.columns = ['Label', 'Count']
        dist['Pct'] = (dist['Count'] / dist['Count'].sum() * 100).round(2)
        col_a, col_b = st.columns([3, 1])
        with col_a:
            color_map = {str(k): v for k, v in CLASS_COLORS.items()}
            fig = px.bar(dist, x='Label', y='Count', color='Label',
                         color_discrete_map=color_map, text='Pct')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside',
                              marker_line_color='rgba(0,0,0,0)')
            fig.update_layout(showlegend=False, height=290, **CHART_BASE)
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.dataframe(dist, use_container_width=True, hide_index=True)

    rule('Data Preview — first 30 rows')
    st.dataframe(df.head(30), use_container_width=True)


def render_predictor(model, tokenizer):
    rule('Input')
    user_input = st.text_area(
        '',
        placeholder='Type or paste any text to classify...',
        height=110,
        label_visibility='collapsed',
    )
    run = st.button('Run inference')

    if run:
        if not user_input.strip():
            st.warning('No input provided.')
            return

        cleaned = clean_text(user_input)
        seq     = tokenizer.texts_to_sequences([cleaned])
        padded  = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        preds   = model.predict(padded, verbose=0)[0]

        class_idx  = int(np.argmax(preds))
        confidence = float(preds[class_idx] * 100)
        sentiment  = LABELS[class_idx]
        clr        = CLASS_COLORS[sentiment]

        rule('Result')
        st.markdown(
            f'<div class="result-banner" style="border-left-color:{clr}">'
            f'<span class="result-label" style="color:{clr}">{sentiment.upper()}</span>'
            f'<span class="result-conf">'
            f'{confidence:.2f}% confidence &nbsp;·&nbsp; '
            f'{len(cleaned.split())} tokens after cleaning'
            f'</span></div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([3, 2])
        with col1:
            rule('Probability Distribution')
            st.plotly_chart(chart_prob_bar(preds), use_container_width=True)
        with col2:
            rule('Raw Scores')
            st.dataframe(pd.DataFrame({
                'Class':       LABELS,
                'Probability': [f'{p:.6f}' for p in preds],
                'Percent':     [f'{p*100:.2f}%' for p in preds],
            }), use_container_width=True, hide_index=True)

            rule('Cleaned Input')
            st.code(cleaned or '(empty after cleaning)', language='text')


def render_footer():
    st.markdown("""
    <div class="dev-wrap">
      <div class="dev-sidebar">
        <img class="dev-avatar"
             src="https://ui-avatars.com/api/?name=Yash+Vasudeva&size=180&background=2E2B29&color=C8973A&bold=true&rounded=false"
             alt="YV" />
        <div class="dev-monogram">Developed by</div>
      </div>
      <div class="dev-body">
        <div class="dev-name">Yash Vasudeva</div>
        <div class="dev-title">Data &amp; AI Professional &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; Deep Learning</div>
        <div class="dev-tagline">Building systems that learn, infer, and explain.</div>
        <div class="dev-links">
          <a class="dev-link" href="https://www.linkedin.com/in/yash-vasudeva/" target="_blank">LinkedIn</a>
          <a class="dev-link" href="https://github.com/yashvasudeva1" target="_blank">GitHub</a>
          <a class="dev-link" href="https://yashvasudeva.vercel.app/" target="_blank">Portfolio</a>
          <a class="dev-link" href="mailto:vasudevyash@gmail.com">Contact</a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title='Sentiment Analysis Studio',
        page_icon=None,
        layout='wide',
        initial_sidebar_state='collapsed',
    )
    st.markdown(CSS, unsafe_allow_html=True)
    ensure_nltk()

    st.markdown("""
    <div class="masthead">
      <div class="masthead-label">Natural Language Processing &nbsp;·&nbsp; Deep Learning</div>
      <h1>Twitter <em>Sentiment</em> Analysis Studio</h1>
      <div class="masthead-sub">
        Bidirectional LSTM &nbsp;·&nbsp; Full training transparency &nbsp;·&nbsp;
        Live inference &nbsp;·&nbsp; Evaluation diagnostics
      </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        model, tokenizer = load_assets()
    except Exception as e:
        st.error(f'Could not load model assets: {e}')
        st.info(
            'Ensure models/sentiment_weights.weights.h5 and models/tokenizer.pickle are present, '
            'and run on Python 3.11/3.12 with TensorFlow installed.'
        )
        return

    tabs = st.tabs([
        'Overview',
        'Training Insights',
        'Evaluation',
        'Model Info',
        'Dataset Info',
        'Predictor',
    ])

    with tabs[0]: render_overview()
    with tabs[1]: render_training()
    with tabs[2]: render_evaluation(model)
    with tabs[3]: render_model_info(model, tokenizer)
    with tabs[4]: render_dataset()
    with tabs[5]: render_predictor(model, tokenizer)

    st.divider()
    render_footer()


if __name__ == '__main__':
    main()
