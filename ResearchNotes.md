# Research Notes: Named Entity Recognition Implementation

## Challenges in NER

1. **Ambiguity**  
   Named entities can be ambiguous. For example, "Apple" could refer to a fruit or the company. Proper context is essential for correct classification.

2. **Boundary Detection**  
   Identifying where a named entity starts and ends can be tricky—especially when entities are nested or split by punctuation. For instance, "President of the United States" vs. "United States" within that phrase.

3. **Contextual Variability**  
   Entities can have multiple meanings depending on context. E.g., "Jordan" could be a person or a country. Accurately capturing context is crucial.

4. **Data Imbalance**  
   Certain entity types may occur more frequently than others (e.g., many person mentions but fewer organization mentions). Imbalanced data can harm performance.

5. **Domain Adaptation**  
   Applying NER to specialized domains (like biomedical text) often requires domain-specific knowledge. Models trained on general-purpose corpora may struggle in these contexts.

6. **Annotation Errors**  
   Manually labeled datasets can contain inconsistencies or mistakes, introducing noise that negatively impacts model training.

7. **Overfitting**  
   With limited or domain-specific data, a large model can easily overfit, failing to generalize to unseen text.

## Data & Modeling Approach

### 1. Data Preparation  
- The pipeline uses Hugging Face Datasets or custom datasets in `data/raw`
- Tokenization and label alignment handle subwords carefully (important for Transformers)
- For domain-specific NER tasks (e.g., addresses, place names), domain knowledge can be integrated similar to specialized entity recognizers

### 2. Modeling  
- Choice of any Transformer backbone (e.g., BERT, RoBERTa)
- Optional CRF layer refines boundary predictions and ensures sequence-level consistency
- Support for parameter-efficient fine-tuning methods (e.g., LoRA, Adapters) for domain adaptation or smaller GPU constraints

### 3. Training  
- Managed by PyTorch Lightning for clarity and reproducibility
- Easy to incorporate advanced callbacks (e.g., early stopping, logging)
- Multiple loss functions (including CRF and specialized objectives) integrated with a uniform interface

### 4. Evaluation  
- Computed with `seqeval` for standard metrics (precision, recall, F1)
- Simple interface to compare performance across multiple models, losses, and domain settings

## Problem Formulation

NER is fundamentally about finding and classifying named entities in unstructured text. Traditional approaches treat it as token classification, where each token is assigned a label indicating the entity type or "outside" (non-entity). However, each domain or dataset introduces unique challenges:

1. **Semantic Overlap**: Some entity types overlap semantically (e.g., "org" vs. "company")
2. **Label Misalignment**: Minor label misalignments can significantly degrade performance (e.g., "B-PER" vs. "B-ORG")
3. **Domain Specificity**: Different domains may require specialized entity types or handling

These challenges motivate our flexible loss objectives and domain-oriented training procedures.

## Advanced Loss Functions Research

### 1. Focal Loss

**Problem Addressed**: Data imbalance in entity classes can lead to poor performance on rarer classes.  
**Objective**: Focal Loss modifies cross-entropy by down-weighting easy examples and focusing on hard ones:
```
FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
```
where `p_t` is the probability of the true class, `α_t` is a balancing factor, and `γ > 0` is the focusing parameter. This effectively upweights underrepresented or harder samples.

### 2. Boundary-aware Loss

**Problem Addressed**: Fine-grained boundary detection is critical for NER.  
**Objective**: A specialized loss that measures the overlap (e.g., Intersection over Union, IoU) between predicted entity spans and gold spans can help the model learn precise boundaries:
```
IoU = | A ∩ B | / | A ∪ B |
```
Incorporating IoU-like metrics into the loss can better align the model's predictions with correct entity boundaries.

### 3. Multi-task Learning Objectives

**Problem Addressed**: Contextual variability & domain adaptation.  
**Objective**: Combine NER with related tasks (POS tagging, entity linking, etc.) under a multi-task framework. Each task contributes to a combined loss:
```
Total Loss = λ1 * Loss_NER + λ2 * Loss_POS + ...
```
where different λ weights govern each task's importance. This helps the model share representations and adapt more robustly.

### 4. Contrastive Loss

**Problem Addressed**: Disambiguating entities that appear similar (e.g., "Jordan" the country vs. the person).  
**Objective**: Create positive and negative pairs of entity contexts and penalize incorrect clustering with a margin-based or distance-based loss:
```
L = (1/N) * Σ [ y_i * max(0, m - D(x_i, x_j)) + (1 - y_i) * D(x_i, x_j) ]
```
This pushes confusing entities apart in embedding space, improving clarity.

### 5. CRF Layer (Conditional Random Fields)

**Problem Addressed**: Dependent or sequential labels in NER (e.g., `I-PER` must follow `B-PER`).  
**Objective**: A CRF can incorporate transition constraints and sequence-level decoding, ensuring more coherent label predictions.

## Innovation: Enhanced Label Smoothing with Embeddings

**Traditional label smoothing** uses a fixed probability (e.g., 0.1) for off-target labels. Our approach **learns label embeddings** to capture *relationships* between entity types:

1. **Label Embeddings**: Each label (e.g., `B-PER`, `I-PER`, `B-ORG`) is assigned a learnable vector.  
2. **Similarity-Based Softening**: The smoothing distribution is computed from similarity between the *gold label* embedding and all other label embeddings (e.g., `B-ORG` might get a higher "similarity score" than `B-LOC`).  
3. **Adaptive Weighting**: A temperature parameter (α) controls the distribution's sharpness, balancing between hard targets and fully softened labels.

**Benefits**:  
- Preserves relationships between labels (makes "mistakes" less penalized when they are semantically close).  
- Improves generalization and domain transfer, inspired by a teacher-student strategy.

**Usage Example**:
```python
from src.losses.embedding_label_smoothing import EmbeddingLabelSmoothing

loss_fn = EmbeddingLabelSmoothing(
    num_labels=9,        # e.g., number of NER tags
    hidden_size=768,     # from your transformer model
    label_emb_dim=32,    # dimension for label embeddings
    smoothing_alpha=10.0 # temperature
)
```

## References & Related Works

1. **ArcGIS `EntityRecognizer` Model**  
   - Provided insight into domain adaptation for specialized entity recognition (e.g., location-based entities).  
   - Showcases advanced fine-tuning and real-world usage scenarios.

2. **Hugging Face Transformers**  
   - For robust, pretrained language models and easy integration with token-classification tasks.

3. **PyTorch Lightning**  
   - For structured, reproducible experiments and flexible training loops.

4. **torchcrf**  
   - Improves entity boundary detection by modeling label dependencies.

5. Research in **label smoothing** & **soft targets**  
   - Inspired by teacher-student methods in knowledge distillation, which guided the label-embedding approach.
