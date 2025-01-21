# NER Using a Transformed LLM (Llama3.2-1B-Instruct)

## 1. Introduction

In this report, we detail our approach to transforming an open-source LLM into a genuine NER model. We briefly summarize:
- The dataset used
- The model architecture and training method
- The evaluation criteria and final results
- Key references and existing tools (e.g., ArcGIS' approach)

The implementation can be run with **`quickstart.ipynb`** and has additional details in **`README.md`**.  
The training and evaluation hyperparameters are configured in **`config/config.yaml`**.

---

## 2. Dataset and Model Selection

### 2.1 Choice of Dataset
- **Dataset Name**: `eriktks/conll2003`
- **Size**: `[PLACEHOLDER for size]`
- **Reason for Selection**: `[PLACEHOLDER for rationale]`

### 2.2 Choice of Model
- **Base LLM**: `Llama3.2-1B-Instruct`
- **Size**: `~1B parameters`
- **Reason for Selection**: `[PLACEHOLDER for rationale]`

---

## 3. Model Adaptation & Training Approach

### 3.1 Architecture and Transformations

- **Original Model Config**:  
  - Base Model: `Llama3.2-1B-Instruct`  
  - Freeze Backbone: `true` or `false` (configurable in `config.yaml`)  
  - `use_crf`: `true`  
  - `peft`: `lora`

- **Token Classification Head**:  
  - A linear classifier of size `(hidden_size, 9)` to predict the 9 NER labels (e.g., O, B-PER, I-PER, etc.).  
  - A CRF layer on top to enforce valid label transitions.

- **Objective Formulation**:  
  - `compound_loss` with specified weights for cross-entropy, CRF, and label smoothing.  
  - The log-likelihood from CRF is combined with cross-entropy and label smoothing in a weighted manner.

### 3.2 Training Procedure
- **Hyperparameters**:  
  - Learning Rate: `5e-5`  
  - Batch Size: `8`  
  - Number of Epochs: `100`  
- **Data Splits**:  
  - Train: `[PLACEHOLDER for train size]`  
  - Validation: `[PLACEHOLDER for val size]`  
  - Test: `[PLACEHOLDER for test size]`  
- **Key Metrics from Training Logs** (around epoch 28):  
  - `train_loss`: `0.3017629384994507`  
  - `val_loss`: `0.26667168736457825`  
  - `val_accuracy`: `0.6238653063774109`  
  - `epoch`: `28.0`  

### 3.3 Challenges & How They Were Addressed
- **Challenge**: `[PLACEHOLDER for challenge 1]`
  - **Approach**: `[PLACEHOLDER for approach 1]`
- **Challenge**: `[PLACEHOLDER for challenge 2]`
  - **Approach**: `[PLACEHOLDER for approach 2]`

---

## 4. Evaluation and Performance

### 4.1 Evaluation Metrics
We report **accuracy, precision, recall, and F1** on the test set.  
*(Below are placeholders; final numbers should be computed.)*

| Metric      | Score         |
|-------------|--------------:|
| Accuracy    | `0.6239`      |
| Precision   | `[PLACEHOLDER]` |
| Recall      | `[PLACEHOLDER]` |
| F1-Score    | `[PLACEHOLDER]` |

### 4.2 Failure Case Analysis
- **Common Misclassifications**:  
  1. `[PLACEHOLDER for misclassification type]`  
  2. `[PLACEHOLDER for misclassification type]`

- **Potential Reasons**: `[PLACEHOLDER for reasons]`  
- **Mitigation Strategies**: `[PLACEHOLDER for improvements]`

---

## 5. Discussion

### 5.1 Why We Chose This Dataset and Model
- `[PLACEHOLDER for reasons to choose conll2003 + Llama3.2-1B]`

### 5.2 Differences in Training vs. Regular LM
- Instead of standard next-token generation, we:
  - Attached a token-classification head with CRF.
  - Used a compound loss combining cross-entropy, CRF negative log-likelihood, and label smoothing.
  - Possibly froze the backbone layers and only updated LoRA parameters plus classifier head.

### 5.3 Limitations & Possible Improvements
- **Limitations**:
  - `[PLACEHOLDER for limitation 1]`
  - `[PLACEHOLDER for limitation 2]`
- **Potential enhancements**:
  - `[PLACEHOLDER for improvement 1]`
  - `[PLACEHOLDER for improvement 2]`

---

## 6. Conclusion

We successfully adapted `Llama3.2-1B-Instruct` for token-level NER classification with a compound loss and CRF head. Early training logs show a `val_accuracy` of about `0.6239` after ~28 epochs, and further improvements may be possible with continued training or partial unfreezing. While some placeholders remain for final metrics, the approach highlights the feasibility of using a smaller Llama-based model for NER. Existing tools such as [ArcGIS Python Mistral LLM integration](https://developers.arcgis.com/python/latest/guide/use-mistral-llm-for-text-classification-and-entity-recognition/) also inspire how generative models can be used for classification tasks.

---

## Appendices (Optional)

- **Code and Notebook**: Submitted as `quickstart.ipynb`
- **Scripts**: `[PLACEHOLDER for any additional scripts or CLI usage]`
