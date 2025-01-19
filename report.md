# NER Using a Transformed LLM (Llama-3.2-1B-Instruct)

## 1. Introduction

In this report, we detail our approach to transforming an open-source LLM into a genuine NER model. We briefly summarize:
- The dataset used
- The model architecture and training method
- The evaluation criteria and final results
- A comparison to alternative approaches

---

## 2. Dataset and Model Selection

### 2.1 Choice of Dataset
- **Dataset Name**: `eriktks/conll2003`
- **Size**: `[14,041 train samples / 3,250 validation samples / 3,453 test samples]` (typical for CONLL-2003)
- **Reason for Selection**: `[A well-known benchmark for NER, moderate size, widely used in research]`

### 2.2 Choice of Model
- **Base LLM**: `unsloth/Llama-3.2-1B-Instruct`
- **Size**: `~1B parameters`
- **Reason for Selection**: `[Demonstrates how a smaller Llama variant can be adapted for token classification]`

---

## 3. Model Adaptation & Training Approach

### 3.1 Architecture and Transformations

- **Original Model Config**:  
  - Base Model: `unsloth/Llama-3.2-1B-Instruct`  
  - Freeze Backbone: `true` (per `config.yaml`, meaning the backbone is kept frozen except for some adapter layers or classifier head)

- **Token Classification Head**:  
  - We added a linear classifier of size `(hidden_size, 9)` to predict NER labels (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC).  
  - `use_crf` is set to `true`, meaning a CRF layer is stacked on top of the logits to enforce valid label transitions.

- **Parameter-Efficient Tuning (PEFT)**:  
  - `peft` is set to `lora`. We insert LoRA adapters into the attention layers for efficient fine-tuning.  
  - Because `freeze_backbone` is `true`, only LoRA parameters and the classifier/CRF layers are updated.

- **Differences vs. Regular LM Training**:
  1. **Objective**: Instead of next-token prediction, we use a token-classification (plus CRF) objective with cross-entropy, label smoothing, and CRF log-likelihood.  
  2. **Loss Function**: We employ a compound loss (`compound_loss`) that combines cross-entropy, CRF negative log-likelihood, and label smoothing in a weighted fashion.

### 3.2 Training Procedure
- **Hyperparameters**:  
  - Learning Rate: `5e-5`  
  - Batch Size: `8`  
  - Number of Epochs: `100`  
  - Loss Function: `compound_loss` with weights  
    - `cross_entropy_weight`: 0.5  
    - `crf_weight`: 0.3  
    - `label_smoothing_weight`: 0.2  

- **Hardware Used**: `Single GPU (1 GPU as specified in config.yaml)`

- **Data Splits** (typical for ConLL-2003):
  - Train: `14,041 samples`
  - Validation: `3,250 samples`
  - Test: `3,453 samples`

### 3.3 Challenges & How They Were Addressed
- **Challenge**: Integrating CRF + LoRA  
  - **Approach**: We carefully modified the forward pass to handle CRF decoding after applying LoRA-based attention updates, ensuring the final logits are dimensionally correct for CRF.
- **Challenge**: Freezing the backbone**  
  - **Approach**: We froze all base model parameters and only allowed LoRA adapters and the classifier layers to update, reducing memory usage and focusing training on the final layers.

---

## 4. Evaluation and Performance

### 4.1 Evaluation Metrics
We report **accuracy, precision, recall, and F1** on the test set.

| Metric      | Score        |
|-------------|-------------:|
| Accuracy    | `[ACCURACY]` |
| Precision   | `[PRECISION]`|
| Recall      | `[RECALL]`   |
| F1-Score    | `[F1]`       |

*(Place your actual scores above. For example, `Accuracy: 97.1%, Precision: 90.5%, Recall: 89.2%, F1: 89.8%`.)*

### 4.2 Failure Case Analysis
- **Common Misclassifications**:  
  1. `[Confusion between B-ORG and B-MISC for certain brand names]`  
  2. `[Error on multi-word location entities with short contexts]`

- **Potential Reasons**: `[Limited context for multi-word entities, partial freezing might hamper domain-specific adaptation for brand names, etc.]`
- **Mitigation Strategies**: `[Increasing training epochs, adjusting LoRA rank, or unfreezing more layers in the backbone]`

---

## 5. Comparison to Other Approaches

### 5.1 Prompt Engineering on ChatGPT/Anthropic

- **Prompt**:
Please identify all named entities in the following text and provide their types using the format: <entity> - <type>

Text: [INSERT SAMPLE SENTENCE HERE]

- **Observed Performance**:
- `[Precision]`: `[VALUE]`
- `[Recall]`: `[VALUE]`
- `[F1]`: `[VALUE]`

*(You can either run a handful of examples or approximate a test set approach. Include the results here if you systematically tested it.)*

### 5.2 Benchmark (Finetuned BERT or Other SOTA)
- **Chosen Baseline**: `bert-base-uncased` (finetuned)
- **Reported Performance**:
- `[Accuracy / F1 / Etc.]`: `[INSERT VALUES FROM PAPER OR YOUR RUN]`
- **Comparison**: 
- `[Our CRF-Llama model is within X% of the BERT baseline, or surpasses it on entity recall, etc.]`

---

## 6. Discussion

### 6.1 Why We Chose This Dataset and Model
- We selected `eriktks/conll2003` because it is a standard, well-understood NER benchmark, and it fits well within the time constraints for experiments.  
- For the model, we used `unsloth/Llama-3.2-1B-Instruct` to demonstrate that a smaller LLaMA variant, combined with parameter-efficient methods like LoRA, can handle NER effectively without massive hardware.

### 6.2 Differences in Training vs. Regular LM
- We trained the model with **compound_loss** for **token classification** rather than next-token generation. This merges cross-entropy on labels, CRF constraints, and label smoothing, focusing on labeling accuracy rather than generative quality.

### 6.3 Limitations & Possible Improvements
- **Limitations**:  
- A 1B-parameter model may still be large for some contexts; training can be slow if not carefully optimized.  
- Freezing the backbone might limit domain adaptation.
- **Potential enhancements**:
1. **Partial unfreezing** of higher layers for potentially higher F1.  
2. **Data augmentation** or multi-task learning with similar tagging tasks.  
3. **Quantization** to reduce memory footprint and speed inference.

---

## 7. Conclusion

We successfully transformed `unsloth/Llama-3.2-1B-Instruct` into a specialized NER model using LoRA-based fine-tuning and a CRF head. Our approach achieved `[F1]` on the `eriktks/conll2003` test set, demonstrating the viability of adapting a generative LLM to a discriminative sequence-labeling task. While freezing the backbone reduced resource usage, future work could explore partial unfreezing for improved performance.

---

## Appendices (Optional)

- **Code and Notebook**: Submitted as `[NOTEBOOK / .IPYNB FILE]`
- **Scripts**: `[ANY SHELL SCRIPTS OR CLI COMMANDS USED]`

