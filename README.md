# Twitter Sentiment Analysis Studio

A compact sentiment analysis studio built around a Bidirectional LSTM pipeline and presented through an editorial-style Streamlit interface.

The application combines four perspectives in one place:
- model internals
- training and evaluation behavior
- dataset quality and class balance
- live inference from raw text

The objective of this project is transparency. Instead of hiding the model behind a single prediction card, the app exposes the training artifacts, asset health, class distributions, and prediction confidence structure in a format that is easy to inspect.

## Project Layout

```
mini projects/
├── app.py
├── README.md
├── data/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
├── models/
│   ├── sentiment_model.h5
│   ├── sentiment_model.keras
│   ├── sentiment_weights.weights.h5
│   └── tokenizer.pickle
├── artifacts/
│   ├── training_history.json
│   └── eval_metrics.json
└── scripts/
    └── save_training_artifacts.py
```

## Folder Intent

### data/
Holds the fixed datasets used for model development and validation:
- `twitter_training.csv`
- `twitter_validation.csv`

Both files are read as four-column records:
1. id
2. entity
3. sentiment
4. text

### models/
Stores model and tokenizer assets required for inference and diagnostics:
- complete model exports (`.h5`, `.keras`)
- weights file used by the app for loading (`sentiment_weights.weights.h5`)
- fitted tokenizer (`tokenizer.pickle`)

### artifacts/
Contains generated numeric traces and evaluation summaries:
- epoch-level training history (`training_history.json`)
- split-level and per-class evaluation metrics (`eval_metrics.json`)

The app falls back to deterministic sample data when these artifact files are missing, which keeps the visual sections operational without breaking the interface.

### scripts/
Utility script location:
- `save_training_artifacts.py` serializes training history and evaluation metrics into the `artifacts/` directory.

## App Surfaces

### Overview
A high-level summary of class space, configuration, and section map.

### Training Insights
Presents loss and accuracy trajectories, best-epoch signals, and epoch tables driven by `artifacts/training_history.json` when available.

### Evaluation
Displays train-test comparison tables, radar overlays, per-class metric bars, and parameter distribution summaries.

### Model Info
Documents architecture details, layer table, preprocessing steps, and model-asset status checks.

### Dataset Info
Focused on the two training datasets in `data/`:
- fixed file-based loading
- sentiment value-count comparison across both datasets
- profile stats and split-wise preview
- direct file download actions for training and validation sets

### Predictor
Accepts free text, applies the same cleaning pipeline used during model preparation, and returns class probabilities plus confidence.

## Data and Labeling Notes

Sentiment labels expected by the model:
- Irrelevant
- Negative
- Neutral
- Positive

Class order is fixed and shared across:
- output softmax mapping
- charts
- probability rendering
- metrics tables

## Preprocessing Summary

Text is normalized through a compact NLTK pipeline:
- lowercase conversion
- tokenization (punkt)
- stopword filtering
- alphabetic token filtering
- lemmatization (WordNet)
- sequence projection through trained tokenizer
- post-padding/truncation to fixed sequence length

## Design Direction

The interface intentionally uses a restrained editorial visual language:
- high-contrast dark base
- amber accent hierarchy
- mono + serif type pairing
- compact chart surfaces with explicit boundaries

The style system keeps visual rhythm consistent across metric strips, tables, and charts while preserving dense informational layouts.

## Reliability Characteristics

- Cached model and tokenizer loading for fast repeat renders.
- Cached NLTK resource initialization.
- Asset health section verifies presence, size, and modified timestamp.
- Dataset section validates required fixed files before rendering downstream analysis.

## Dependencies (Logical)

Core runtime stack:
- streamlit
- tensorflow / keras
- numpy
- pandas
- nltk
- plotly
- scikit-learn (artifact script)

## Closing Note

This repository is structured as a practical model-observability mini product rather than a notebook dump. The folder split reflects that intent: data, model assets, generated artifacts, and supporting scripts each have a clear boundary and role.