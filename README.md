# Disaster Tweets Classification

A binary text classification project that predicts whether a tweet describes a real disaster. It compares a pooling baseline against LSTM and GRU models (unidirectional and bidirectional) using pre-trained **GloVe** word embeddings.

## Dataset

Kaggle's *Natural Language Processing with Disaster Tweets* data (`train.csv`), using these columns:

| Column | Description |
|---|---|
| `text` | The tweet |
| `target` | `1` = disaster, `0` = not a disaster (renamed to `label` in the script) |

You also need the pre-trained GloVe vectors `glove.6B.100d.txt` from the [Stanford GloVe page](https://nlp.stanford.edu/projects/glove/).

Neither file is included in this repository. Update the two file paths in the script to point to your copies.

## Pipeline

1. **Cleaning**: drop duplicate tweets and missing values, lowercase, remove punctuation, digits and extra whitespace.
2. **Split**: 90% train / 10% test (`random_state=1000`).
3. **Tokenisation**: Keras `Tokenizer` fitted on the training set, with sequences padded to the longest training sequence.
4. **Embeddings**: a 100-dimensional embedding matrix built from GloVe 6B (about 62% of the vocabulary is found in GloVe).
5. **Training**: each model uses `adam`, binary cross-entropy, up to 20 epochs, batch size 32, a 10% validation split, and early stopping on validation loss (patience 3).
6. **Evaluation**: accuracy, precision, recall and F1 on train and test sets, with training curves and train-vs-test bar charts.

## Models

| Model | Architecture |
|---|---|
| Baseline | Embedding, global max pooling, Dense(128), sigmoid |
| Uni-LSTM | Embedding, LSTM(32), Dropout(0.1), sigmoid |
| Bi-LSTM | Embedding, Bidirectional LSTM(32), Dropout(0.1), sigmoid |
| Uni-GRU | Embedding, GRU(32), Dropout(0.1), sigmoid |
| Bi-GRU | Embedding, Bidirectional GRU(24), Dropout(0.1), sigmoid |

## Results (test set, as recorded in the script)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Shallow RNN | 0.8176 | 0.8474 | 0.6806 | 0.7549 |
| Uni-LSTM | 0.8189 | 0.8919 | 0.6387 | 0.7444 |
| Bi-LSTM | 0.8256 | 0.8841 | 0.6645 | 0.7587 |
| Uni-GRU | 0.8162 | 0.8386 | 0.6871 | 0.7553 |
| Bi-GRU | 0.8229 | 0.8149 | 0.7387 | 0.7750 |

Bi-GRU has the best recall and F1, while Uni-LSTM has the best precision.

## Getting Started

### Requirements

- Python 3.9+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- tensorflow (Keras)

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Run

1. Download `train.csv` and `glove.6B.100d.txt`.
2. In the script, replace the `/content/drive/MyDrive/CSE427/Project/...` paths with your local paths.
3. Run:

```bash
python disaster_analysis.py
```

The script was exported from Google Colab, so running it cell by cell in Colab or Jupyter also works.

## Project Structure

```
disaster-tweets/
├── disaster_analysis.py
└── README.md
```

## Known Limitations

- The script is a notebook export and relies on global state, so cells must run in order.
- File paths are hard-coded for Google Drive.
- The final comparison plots use metric values entered by hand from earlier runs, so they will not update when you retrain.
- Metrics come from a single random split, so small differences between models (under about 1%) may not be meaningful.

## Authors

_Add your name(s) and course/university information here._

## License

_Add a license (for example MIT) before publishing._
