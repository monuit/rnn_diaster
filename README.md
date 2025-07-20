# NLP Disaster Tweets Classification with RNNs

A comprehensive machine learning project that implements and compares various Recurrent Neural Network (RNN) architectures to classify tweets as disaster-related or non-disaster tweets using the Kaggle "Natural Language Processing with Disaster Tweets" dataset.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Results](#results)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## 🎯 Overview

This project addresses the challenge of automatically identifying disaster-related tweets from social media data. During emergencies, Twitter serves as a valuable real-time information source for disaster relief organizations and news agencies. However, distinguishing between tweets reporting actual disasters versus metaphorical usage of disaster-related terms (e.g., "the party was fire" vs. "there's a fire downtown") requires sophisticated natural language processing.

The project implements and compares four different RNN architectures:
- **Vanilla RNN**: Baseline simple recurrent network
- **LSTM**: Long Short-Term Memory networks for capturing long-range dependencies
- **Bidirectional LSTM**: Processes text in both forward and backward directions
- **Deep LSTM**: Multi-layer stacked architecture for enhanced representation learning

## 📊 Dataset

**Source**: [Kaggle - Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

**Dataset Characteristics**:
- **Training Set**: 7,613 tweets with target labels
- **Test Set**: 3,263 tweets for predictions
- **Class Distribution**: 43% disaster tweets, 57% non-disaster tweets
- **Text Length**: Most tweets contain 9-11 words after preprocessing
- **Sequence Length**: 90% of tweets contain ≤15 words (used as max sequence length)

**Data Fields**:
- `id`: Unique identifier for each tweet
- `text`: The actual tweet content
- `location`: Tweet location (may be blank)
- `keyword`: Keyword from the tweet (may be blank)
- `target`: Binary label (1 = disaster, 0 = non-disaster)

## 🏗️ Project Structure

```
├── week4_rnn_disaster_tweets.ipynb    # Main Jupyter notebook
├── nlp-getting-started/               # Dataset directory
│   ├── train.csv                      # Training data
│   ├── test.csv                       # Test data
│   └── sample_submission.csv          # Sample submission format
├── kaggle.csv                         # Final predictions
└── README.md                          # This file
```

## 🧠 Model Architectures

### 1. Vanilla RNN
- **Architecture**: Embedding → SimpleRNN(64) → Dense(1, sigmoid)
- **Purpose**: Baseline model for comparison
- **Performance**: 77.9% validation accuracy

### 2. LSTM
- **Architecture**: Embedding → LSTM(64) → Dense(1, sigmoid)
- **Features**: Gating mechanisms for long-term dependencies
- **Performance**: 80.5% validation accuracy

### 3. Bidirectional LSTM
- **Architecture**: Embedding → Bidirectional(LSTM(64)) → Dense(1, sigmoid)
- **Features**: Processes text in both directions for context
- **Performance**: 77.8% validation accuracy

### 4. Deep LSTM (Best Model)
- **Architecture**: Embedding → LSTM(128) → LSTM(64) → Dense(64, relu) → Dense(1, sigmoid)
- **Features**: Multiple layers for enhanced representation
- **Performance**: 80.4% validation accuracy

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook
- Kaggle API credentials (for dataset download)

### Required Libraries
```bash
pip install numpy pandas matplotlib seaborn
pip install nltk tensorflow scikit-learn
pip install gensim kaggle
```

### NLTK Data Downloads
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### Dataset Setup
1. Install Kaggle API: `pip install kaggle`
2. Configure Kaggle credentials
3. Run the notebook - it will automatically download the dataset

## 💻 Usage

### Running the Complete Analysis
```bash
jupyter notebook week4_rnn_disaster_tweets.ipynb
```

### Key Notebook Sections:
1. **Data Loading & EDA**: Dataset exploration and visualization
2. **Text Preprocessing**: Cleaning, tokenization, and sequence preparation
3. **Model Training**: Implementation of four RNN architectures
4. **Performance Comparison**: Accuracy and loss analysis
5. **Optimization Experiments**: Optimizer and regularization testing
6. **Predictions**: Generate final Kaggle submission

### Text Preprocessing Pipeline
```python
def clean_text(text):
    text = text.lower()                          # Lowercase conversion
    text = re.sub(r'\W', ' ', text)             # Remove special characters
    text = re.sub(r'\s+', ' ', text)            # Normalize whitespace
    # Lemmatization and stopword removal
    text = ' '.join([lemmatizer.lemmatize(word) 
                    for word in text.split() 
                    if word not in stop_words])
    return text
```

## 📈 Results

### Model Performance Comparison

| Model | Validation Accuracy | Validation Loss | Training Time |
|-------|-------------------|-----------------|---------------|
| Vanilla RNN | 77.9% | 0.48 | ~13s/epoch |
| LSTM | 80.5% | 0.44 | ~1s/epoch |
| Bidirectional | 77.8% | 0.49 | ~5s/epoch |
| **Deep LSTM** | **80.4%** | **0.46** | ~1s/epoch |

### Optimization Experiments
- **Adam vs RMSprop**: Adam optimizer slightly outperformed RMSprop
- **L2 Regularization**: Minimal impact with λ = 0.001 and λ = 0.0001
- **Early Stopping**: Prevented overfitting across all models

## 🔍 Key Findings

1. **Architecture Impact**: LSTM and Deep models significantly outperformed vanilla RNN
2. **Bidirectional Limitation**: Didn't improve performance, possibly due to short sequence lengths
3. **Deep Architecture**: Best overall performance with increased model capacity
4. **Regularization**: Models showed good generalization without explicit regularization
5. **Optimizer Choice**: Adam remained superior to RMSprop for this task
6. **Data Characteristics**: Balanced dataset with manageable sequence lengths ideal for RNN approaches

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras
- **NLP**: NLTK, Gensim Word2Vec
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Environment**: Jupyter Notebook
- **Dataset**: Kaggle API

## 📚 Learning Outcomes

This project demonstrates:
- Implementation of various RNN architectures
- Text preprocessing techniques for neural networks
- Comparative analysis of deep learning models
- Hyperparameter tuning and optimization strategies
- Real-world NLP application for disaster response

## 🤝 Contributing

This project was developed as part of a peer-graded assignment. For improvements or questions:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed descriptions

## 📄 License

This project is for educational purposes. Dataset courtesy of Kaggle's "Natural Language Processing with Disaster Tweets" competition.

---

**Note**: This project achieves competitive performance on the disaster tweet classification task, demonstrating the effectiveness of deep RNN architectures for short-text classification in emergency response scenarios.