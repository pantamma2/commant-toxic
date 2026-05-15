#  Comment Toxicity Classifier

A deep learning model that detects toxic comments across 6 categories using a Bidirectional LSTM neural network — with a live Gradio web interface for real-time predictions.

---

##  Demo

**Input:** `"You freaking suck! I am going to hit you."`

**Output:**
```
toxic: True
severe_toxic: False
obscene: True
threat: True
insult: True
identity_hate: False
```

---

##  How It Works

```
Raw Comment Text
      │
      ▼
TextVectorization (vocab size: 200,000 | sequence length: 1800)
      │
      ▼
Embedding Layer (32 dimensions)
      │
      ▼
Bidirectional LSTM (32 units, tanh activation)
      │
      ▼
Fully Connected Layers (128 → 256 → 128)
      │
      ▼
Output Layer (6 neurons, sigmoid activation)
      │
      ▼
6 Toxicity Scores (threshold: 0.5)
```

---

##  Output Categories

| Label | Description |
|---|---|
| `toxic` | General toxic language |
| `severe_toxic` | Highly aggressive or abusive |
| `obscene` | Obscene or vulgar content |
| `threat` | Direct threats of harm |
| `insult` | Personal insults |
| `identity_hate` | Hate based on identity (race, gender, religion etc.) |

---

##  Tech Stack

| Component | Technology |
|---|---|
| Model | TensorFlow / Keras |
| Architecture | Bidirectional LSTM |
| Text Processing | TextVectorization (Keras) |
| Data Pipeline | tf.data (cache, shuffle, batch, prefetch) |
| Evaluation | Precision, Recall, CategoricalAccuracy |
| UI | Gradio |
| Dataset | Jigsaw Toxic Comment Classification (Kaggle) |

---

##  Model Architecture

```
Layer                    Output Shape        Parameters
─────────────────────────────────────────────────────
Embedding                (None, 1800, 32)    6,400,032
Bidirectional LSTM       (None, 64)          16,896
Dense (relu)             (None, 128)         8,320
Dense (relu)             (None, 256)         33,024
Dense (relu)             (None, 128)         32,896
Dense (sigmoid)          (None, 6)           774
─────────────────────────────────────────────────────
Total params: ~6.5M
```

---

##  Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/pantamma2/commant-toxic.git
cd commant-toxic
```

### 2. Install dependencies
```bash
pip install tensorflow pandas numpy gradio matplotlib
```

### 3. Download the dataset
- Go to: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- Download `train.csv` and place it in the project root

### 4. Train the model
```bash
# Open and run the Jupyter notebook
jupyter notebook Toxicity.ipynb
```
This generates `toxicity.h5`

### 5. Launch the Gradio UI
```python
import gradio as gr
interface.launch(share=True)
```
Opens a live web interface at `http://localhost:7860`

---

##  Project Structure

```
comment-toxicity-classifier/
├── Toxicity.ipynb        # Full training notebook
├── toxicity.h5           # Saved model
├── train.csv             # Kaggle dataset (download separately)
└── README.md
```

---

##  Evaluation Metrics

Evaluated on the test set (10% of data):

| Metric | Score |
|---|---|
| Precision | Run notebook to see result |
| Recall | Run notebook to see result |
| Categorical Accuracy | Run notebook to see result |

> Add your actual scores here after running the notebook!

---

##  Future Improvements

- [ ] Replace LSTM with BERT / DistilBERT for better accuracy
- [ ] Add confidence score visualization
- [ ] Deploy on Hugging Face Spaces
- [ ] Add multi-language support

---

## 👤 Author

**Suryanarayana Murthy Chilukuri**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/suryanarayana-murthy-chilukuri-a88107235)
[![GitHub](https://img.shields.io/badge/GitHub-pantamma2-black?logo=github)](https://github.com/pantamma2)
