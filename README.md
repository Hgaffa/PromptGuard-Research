# 🔬 PromptGuard Research

This repository contains the complete research, experimentation, and model development process for **PromptGuard** - a production-ready library for detecting malicious LLM prompts and prompt injection attacks.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Note**: This is the research repository. For the production-ready Python package, see [promptguard](https://github.com/Hgaffa/promptguard).

---

## 📊 Project Overview

PromptGuard is a machine learning system designed to detect and prevent prompt injection attacks on Large Language Models (LLMs). This repository documents the complete journey from raw data to a production model achieving **97.5% F1-score**.

### Final Model Performance

| Metric | Score |
|--------|-------|
| **F1-Score** | 0.975 |
| **ROC-AUC** | 0.994 |
| **Recall** | 97.24% |
| **Precision** | 97.77% |
| **False Negative Rate** | 2.76% |
| **Inference Speed** | ~13ms per prompt (GPU) |

### Dataset
- **Total Samples**: 40,000 prompts
- **Distribution**: 50% benign, 50% malicious (perfectly balanced)
- **Languages**: 96% English, 4% other
- **Split**: 70% train / 15% validation / 15% test

---

## 🎯 Research Sessions Overview

### Exploratory Data Analysis
**Goal**: Understand the dataset characteristics and identify patterns

**Key Findings**:
- Perfect 50/50 class balance (no resampling needed)
- Malicious prompts are 8x longer on average (503 vs 63 characters)
- 17% of malicious prompts contain special characters (vs 0.3% benign)
- Common malicious keywords: "ignore", "forget", "bypass", "previous"
- 96% of prompts are in English

**Deliverables**:
- Class distribution analysis
- Length distribution visualizations
- Pattern identification
- Data quality assessment

---

### Text Preprocessing & Feature Engineering
**Goal**: Clean data and create meaningful features for modeling

**Features Engineered** (21 total):

**Length Features (2)**:
- `prompt_length`: Character count
- `word_count`: Word count

**Character-Level Features (7)**:
- `special_char_count`, `digit_count`, `uppercase_count`
- `char_entropy`: Text randomness measure
- `special_char_ratio`, `digit_ratio`, `uppercase_ratio`

**Linguistic Features (3)**:
- `avg_word_length`: Average word length
- `lexical_diversity`: Vocabulary richness
- `sentence_count`: Number of sentences

**Security Features (6)**:
- `instruction_keyword_count`: "ignore", "bypass", etc.
- `context_keyword_count`: "previous", "above", etc.
- `jailbreak_keyword_count`: "DAN", "pretend", etc.
- `total_injection_keywords`: Combined count
- `has_base64`: Base64 encoding detection
- `has_hex`: Hex encoding detection

**URL/Email Features (2)**:
- `url_count`, `email_count`

**Code Pattern Features (1)**:
- `has_code_block`: Markdown/code block detection

**Deliverables**:
- Cleaned text data
- 21 engineered features
- Train/val/test splits (70/15/15)
- Feature correlation analysis

---

### Baseline Models
**Goal**: Establish performance benchmarks with simple models

**Models Trained**:
1. **Logistic Regression + TF-IDF** ⭐
   - F1: 0.9504 | AUC: 0.9839
   - **Winner**: Best baseline
   
2. **Random Forest + Features**
   - F1: 0.9341 | AUC: 0.9654

**Key Insights**:
- TF-IDF (5,000 features) beats hand-crafted features (21 features)
- Text representation matters more than model complexity
- Logistic Regression is surprisingly strong for text classification
- Top words for malicious: "forget", "ignore", "disregard"

**Deliverables**:
- Two baseline models
- Performance benchmarks
- Feature importance analysis
- Misclassification analysis

---

### Advanced Models - Gradient Boosting
**Goal**: Test if gradient boosting beats logistic regression

**Models Trained**:
1. **XGBoost**: F1: 0.9352 | AUC: 0.9661
2. **LightGBM**: F1: 0.9332 | AUC: 0.9667
3. **XGBoost (Tuned)**: F1: 0.9350 | AUC: 0.9660

**Surprising Result**: ❌ Gradient boosting **did NOT beat** Logistic Regression!

**Why Logistic Regression Won**:
- TF-IDF creates sparse, high-dimensional features
- Linear models excel at sparse text data
- Trees struggle with feature sparsity
- 5,000 TF-IDF features > 21 aggregated features

**Key Lessons**:
- More complex ≠ better
- Feature representation > model sophistication
- Always validate assumptions empirically

**Deliverables**:
- Three gradient boosting models
- Hyperparameter tuning results
- Comprehensive model comparison
- Lessons on when to use tree-based models

---

### Transformer Model - DistilBERT
**Goal**: Test if deep learning can beat traditional ML

**Model**: DistilBERT (66M parameters, fine-tuned for 3 epochs)

**Results**: 🏆 **WINNER!**
- **F1-Score**: 0.9751 (+0.0246 over LR)
- **ROC-AUC**: 0.9943 (+0.0103 over LR)
- **Recall**: 97.24% (+4.77% over LR)
- **FNR**: 2.76% (63% reduction vs LR!)

**Training**:
- Time: 19.9 minutes on Tesla T4 GPU
- Hardware: GPU recommended
- Epochs: 3 (early stopping at epoch 2)

**Why DistilBERT Won**:
- **Contextual understanding**: "ignore warning" vs "ignore instructions"
- **Semantic similarity**: Detects paraphrased attacks
- **Pre-training**: Leverages 16GB of text knowledge
- **Attention mechanism**: Focuses on important tokens

**Trade-offs**:
- ✅ Better accuracy (+2.5% F1)
- ✅ Much better recall (+4.77%)
- ✅ Lower false negative rate (-63%)
- ❌ 25x slower inference (12.71ms vs 0.5ms)
- ❌ Larger model size (250MB vs 20MB)

**Deliverables**:
- Fine-tuned DistilBERT model
- Training curves and metrics
- Speed vs accuracy analysis
- Token length distribution analysis

---

### Final Validation & Model Selection
**Goal**: Validate on test set and select production model

**Test Set Results** (held-out, never seen during development):
- F1-Score: 0.975 ✓
- ROC-AUC: 0.994 ✓
- Validation-Test gap: <0.001 ✓ (excellent generalization)

**Threshold Optimization**:
Three recommended thresholds for different use cases:

1. **Maximum Security** (threshold: 0.35)
   - FNR: 1.8% (catch 98.2% of attacks)
   - Use case: High-security applications

2. **Balanced** (threshold: 0.50)
   - F1: 0.975 (optimal balance)
   - Use case: General production

3. **User Experience** (threshold: 0.65)
   - FPR: 0.9% (minimal false positives)
   - Use case: User-facing applications

**Ensemble Exploration**:
Tested combining DistilBERT + Logistic Regression:
- Simple average: F1 = 0.973
- Weighted (70/30): F1 = 0.974
- Hybrid (LR screens, BERT decides): F1 = 0.974
- **Result**: DistilBERT alone is best (0.975)

**Final Model Selection**: ✅ **DistilBERT** (solo)

**Deliverables**:
- Test set evaluation
- Calibration analysis
- Optimal thresholds
- Ensemble comparison
- Error analysis
- Deployment documentation

---

## 🔬 Model Comparison Summary

| Model | F1-Score | ROC-AUC | Recall | FNR | Inference Speed |
|-------|----------|---------|--------|-----|-----------------|
| **DistilBERT** 🏆 | **0.9751** | **0.9943** | **97.24%** | **2.76%** | 12.71ms |
| Logistic Regression | 0.9504 | 0.9839 | 92.47% | 7.53% | ~0.5ms |
| XGBoost (Tuned) | 0.9350 | 0.9660 | 88.96% | 11.04% | ~2ms |
| XGBoost | 0.9352 | 0.9661 | 89.03% | 10.97% | ~2ms |
| Random Forest | 0.9341 | 0.9654 | 88.86% | 11.14% | ~1ms |
| LightGBM | 0.9332 | 0.9667 | 88.79% | 11.21% | ~2ms |

**Key Takeaway**: DistilBERT achieves the best performance across all metrics, with a 63% reduction in false negative rate compared to the best baseline.

---

## 📈 Key Research Insights

### 1. Feature Engineering vs Learned Representations
- Hand-crafted features (21): Good but limited
- TF-IDF (5,000): Better - captures more nuance
- Transformer embeddings: Best - contextual understanding

**Winner**: Let the model learn features (transformers)

### 2. Model Complexity Sweet Spot
- Too simple (single features): Poor performance
- Just right (TF-IDF + Logistic Regression): Strong baseline
- Complex but wrong approach (gradient boosting): Doesn't help for text
- Complex and right approach (transformers): Best performance

### 3. The Importance of Recall in Security
For security applications, **False Negative Rate is critical**:
- Missing 1 in 36 attacks (DistilBERT, 2.76% FNR) ✅
- vs Missing 1 in 13 attacks (Logistic Regression, 7.53% FNR)
- vs Missing 1 in 9 attacks (Gradient Boosting, 11% FNR)

**The 63% reduction in FNR justifies the 25x slowdown.**

### 4. Dataset Quality Matters
- Suspected label noise in ~5-10% of "malicious" examples
- Models still achieved high performance despite noise
- Robust models can handle some label imperfection

### 5. Speed vs Accuracy Trade-off
- Logistic Regression: Fast but misses attacks
- DistilBERT: Slower but catches attacks
- **Conclusion**: For security, accuracy > speed (within reason)

---

## 🚀 Production Deployment

The final DistilBERT model has been:
- ✅ Uploaded to HuggingFace Hub: [arkaean/promptguard-distilbert](https://huggingface.co/arkaean/promptguard-distilbert)
- ✅ Packaged as a Python library: [promptguard](https://github.com/YOUR_USERNAME/promptguard)
- ✅ Optimized for production use
- ✅ Documented with deployment guides

### Quick Start (Production Package)
```python
# Install production package
pip install promptguard

# Use the model
from promptguard import PromptGuard

guard = PromptGuard()
result = guard.analyze("Ignore all previous instructions")
print(result.is_malicious)  # True
```

---

## 🔗 Related Resources

- **Production Package**: [promptguard](https://github.com/Hgaffa/promptguard)
- **Model on HuggingFace**: [arkaean/promptguard-distilbert](https://huggingface.co/arkaean/promptguard-distilbert)
- **Demo Application**: [promptguard-app](https://github.com/Hgaffa/promptguard-app) (coming soon)
