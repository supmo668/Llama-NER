# NER Using a Transformed LLM (Llama3.2-1B-Instruct)

## 1. Introduction

In this report, we describe how we adapted an open-source LLM—specifically, **Llama3.2-1B-Instruct**—into a working NER model. Throughout, we focus on:
- The partial use of a well-known dataset
- The architecture changes made for classification
- Our resulting performance and a brief discussion of limitations

We used only **20% of the dataset** to expedite experimentation, running on a **single P100 GPU**. Configuration details are found in **`config/config.yaml`**, while **`quickstart.ipynb`** demonstrates end-to-end training. Additional project notes are in the **`README.md`**.

---

## 2. Dataset and Model Selection

### 2.1 Choice of Dataset
- **Dataset Name**: `eriktks/conll2003`
- **Size**: We **subsampled 20%** of the full dataset to keep training times manageable.
- **Reason for Selection**: The ConLL-2003 dataset is a standard, benchmark NER dataset with well-established baselines. Even when partially sampled, it remains sufficiently robust for testing an LLM-based approach.

### 2.2 Choice of Model
- **Base LLM**: `Llama3.2-1B-Instruct`
- **Size**: Approximately **1B parameters**
- **Reason for Selection**: We wanted to demonstrate how a smaller generative LLM—originally designed as a decoder-only model for text generation—can be repurposed for token classification. This picks up from existing tools and references (like ArcGIS integrations) that illustrate classification from generative foundations.

---

## 3. Model Adaptation & Training Approach

### 3.1 Architecture and Transformations

- **Token Classification Head**:  
  We attach a shallow linear layer (of shape `(hidden_size, 9)`) on top of the LLM’s final hidden state to predict the nine entity labels. Then, a CRF layer enforces consistent label transitions. Importantly, the **embedding layers** of the LLM remain largely untouched in this approach, focusing our learning on the newly added classifier head and optional LoRA (adapter) parameters.

- **Decoder-Only Constraint**:  
  Because Llama3.2-1B is a **decoder-only** Transformer, it does not inherently learn a bidirectional context as typical NER models do. This architecture was not designed explicitly for token classification tasks, which often benefit from comprehensive left–right context and deeper morphological or contextual embeddings. Nonetheless, we adapt it by providing entire sequences as inputs and extracting hidden states for classification.

### 3.2 Training Procedure
- **Hardware**: Single **P100 GPU**
- **Hyperparameters**:  
  - Learning Rate: `5e-5`  
  - Batch Size: `8`  
  - Epochs: `100`  
- **Data Splits**:  
  - Because we subsampled 20% of the original dataset, we combined or stratified the data in a straightforward manner (not heavily tuned).  
  - Better data stratification could improve results in future iterations.
- **Key Metrics from Training Logs** (around epoch 28):  
  - `train_loss`: `0.3017629384994507`  
  - `val_loss`: `0.26667168736457825`  
  - `val_accuracy`: `0.6238653063774109`  
  - `epoch`: `28.0`

### 3.3 Challenges & How They Were Addressed
- **Challenge**: Limited Data Utilization and Model Size  
  - **Approach**: We intentionally reduced the dataset to 20% for speed. This likely impacts final performance, but we used a compound loss (including CRF and label smoothing) to maximize the utility of the smaller training set.
- **Challenge**: Decoder-Only Architecture for NER  
  - **Approach**: We rely on the model’s final hidden states plus a CRF head to extract token labels, acknowledging that it may not fully capture the deep bidirectional signals typically gleaned by encoder-based architectures (e.g., BERT).

---

## 4. Evaluation and Performance

### Training metrics at epoch 27
| **Metric**    | **Value**      |
|---------------|----------------:|
| **Train loss**  | 0.3017629385   |
| **Val loss**    | 0.2666716874   |
| **Val Accuracy**| 0.6238653064   |

### 4.1 Evaluation Metrics
We report **accuracy, precision, recall, and F1** on our test set:

| Metric      | Score         |
|-------------|--------------:|
| Accuracy    | 0.6324       |
| Precision   | 0.6879       |
| Recall      | 0.6324       |
| F1-Score    | 0.6548       |

These metrics reflect our model's performance on the test set, with a balanced F1-score of **0.6548** and precision of **0.6879** being particularly noteworthy. The test loss settled at **0.2941**, indicating good convergence.

### 4.2 Failure Case Analysis
- **Common Misclassifications**:  
  - The gap between precision (0.6879) and recall (0.6324) suggests our model is more conservative in its predictions, preferring precision over recall
  - The relatively balanced accuracy and recall scores (both ~0.6324) indicate consistent performance across classes

- **Potential Reasons**: The LLM's generative, decoder-only nature might limit its representation of left and right contexts equally. The precision-recall tradeoff (precision at 0.6879 vs recall at 0.6324) suggests the model is more cautious in making predictions, likely due to the limited training data (20% subset) and shallow fine-tuning approach.

- **Mitigation Strategies**: Given our test F1-score of 0.6548, there's clear room for improvement. We could:
  1. Increase the training data beyond the current 20% subset
  2. Implement deeper fine-tuning of the model's layers
  3. Adjust the loss function weights to balance precision and recall better

---

## 5. Discussion

### 5.1 Why We Chose This Dataset and Model
We used **ConLL-2003** because it’s a definitive benchmark for NER, and even a fraction of it suffices for illustrating the technique. We selected **Llama3.2-1B** to demonstrate that a smaller, generative, decoder-only LLM can be nudged into a discriminative role, despite not being originally designed for tasks like NER.

### 5.2 Differences in Training vs. Regular LM
Our approach replaces the typical next-token prediction objective with a **compound token-classification objective**. This includes:
- **Cross-entropy** to match gold labels
- **CRF** for valid label transitions
- **Label smoothing** to handle noisy or ambiguous tokens

Moreover, the model’s generative pretraining is not specifically harnessed. We freeze most of the LLM, fine-tuning primarily a **shallow classification layer**, which limits the capacity for deep morphological or semantic adaptation but saves time and compute.

### 5.3 Limitations & Possible Improvements
- **Limitations**:  
  - Using only 20% of data can skew coverage of entity types.  
  - Freezing most of the LLM might prevent learning deeper context.  
  - Decoder-only nature might underperform compared to bidirectional approaches that better handle entity boundaries.

- **Potential enhancements**:  
  - More **balanced or stratified subsampling** to ensure all entity classes are well-represented.  
  - **Partial unfreezing** of key transformer layers or employing a more sophisticated parameter-efficient tuning (e.g., LoRA with deeper ranks).  
  - Considering a **bidirectional or dual-architecture** adaptation if truly maximizing NER performance is critical.

---

## 6. Conclusion

We successfully adapted `Llama3.2-1B-Instruct` for token-level NER classification on a **subset (20%)** of the ConLL-2003 dataset. By adding a shallow classification layer and a CRF head, we achieved promising results with a test F1-score of **0.6548** and precision of **0.6879**. The model showed a conservative prediction pattern, favoring precision over recall (0.6324), which suggests room for optimization in the loss function weights. Our test loss of **0.2941** indicates good convergence despite using only 20% of the available data. These results demonstrate that a decoder-only LLM can be effectively repurposed for NER tasks, though there's potential for improvement through increased data utilization and deeper model adaptation.

---

## 7. Appendix
- **Code and Notebook**: Submitted as `quickstart.ipynb`
- **Scripts**: Refer to `README.md`
