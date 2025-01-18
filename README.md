# NER Pipeline

A comprehensive Named Entity Recognition (NER) pipeline using **PyTorch Lightning** and **Transformers**. This project provides a modular and extensible framework for training and evaluating NER models with optional CRF layers and advanced loss functions. In addition, it explores a cutting-edge **embedding-based label smoothing** approach—motivated by the idea that some entity types are more semantically related than others and should be treated accordingly during training.

---

## Monitoring Training and Evaluation

To monitor the training and evaluation progress, you can use TensorBoard. The logs are stored in the directory specified in the configuration file under `training.log_dir`.

### Start TensorBoard

```bash
./scripts/tensorboard.sh
```

This will launch TensorBoard, and you can view the logs by navigating to `http://localhost:6006` in your web browser.

## Configuring Log and Checkpoint Locations

You can configure the locations for logs and checkpoints in the `config/config.yaml` file:

```yaml
training:
  log_dir: "lightning_logs"  # Directory for storing logs
  checkpoint_dir: "checkpoints"  # Directory for storing model checkpoints
```

Adjust these paths as needed to suit your project structure.

---

## Intuition & Problem Formulation

NER is about **finding and classifying named entities** (e.g., people, organizations, locations) in unstructured text. Traditional approaches treat the task as **token classification**: each token (or sub-token) is assigned a label indicating the entity type or "outside" (non-entity). For example:
```
Barack  B-PER
Obama   I-PER
was     O
born    O
in      O
Hawaii  B-LOC
```
In modern systems, **Transformer models** (e.g., BERT, Llama, Mistral) or specialized frameworks like the **ArcGIS `EntityRecognizer`** model are fine-tuned to this task. However, each domain or dataset can introduce unique nuances:
- Some entity types overlap semantically (e.g., "org" vs. "company").
- Minor label misalignments significantly degrade performance (e.g., "B-PER" vs. "B-ORG").

Hence, **flexible loss objectives** and domain-oriented training procedures become vital. This project embraces **PyTorch Lightning** for structured training, advanced losses (e.g., CRF, embedding-based label smoothing), and a flexible approach to data ingestion.

---

## Challenges in NER

1. **Ambiguity**  
   Named entities can be ambiguous. For example, "Apple" could refer to a fruit or the company. Proper context is essential for correct classification.

2. **Boundary Detection**  
   Identifying where a named entity starts and ends can be tricky—especially when entities are nested or split by punctuation. For instance, “President of the United States” vs. “United States” within that phrase.

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

---

## Intelligent Loss Functions & Objectives

To address the above challenges, we can leverage **advanced loss functions** and **objectives** tailored to NER. These methods add structure, focus, or domain-specific insight into the training process.

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
Incorporating IoU-like metrics into the loss can better align the model’s predictions with correct entity boundaries.

### 3. Multi-task Learning Objectives

**Problem Addressed**: Contextual variability & domain adaptation.  
**Objective**: Combine NER with related tasks (POS tagging, entity linking, etc.) under a multi-task framework. Each task contributes to a combined loss:
```
Total Loss = λ1 * Loss_NER + λ2 * Loss_POS + ...
```
where different λ weights govern each task’s importance. This helps the model share representations and adapt more robustly.

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

---

## Features

- **Modular project structure** with clear separation of concerns  
- **Support for various transformer models** from [Hugging Face](https://huggingface.co/)  
- **Optional CRF layer** for improved entity boundary detection  
- **Advanced loss functions** for better training:
  - **Enhanced Label Smoothing with Embeddings**: Learns semantic relationships between labels  
  - **Focal Loss** for handling class imbalance  
  - **Dice Loss** for improved performance on imbalanced datasets  
  - **Compound Loss** for combining multiple loss functions  
- **PyTorch Lightning** for structured and scalable training  
- **Comprehensive evaluation** metrics using [seqeval](https://github.com/chakki-works/seqeval)  
- **CLI interface** for easy pipeline execution  
- **Configurable hyperparameters** via YAML  

---

## Project Structure

```
my_ner_project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/         # for storing raw data
│   └── processed/   # for storing processed data
├── config/
│   └── config.yaml  # project configs
├── scripts/
│   └── run.sh       # convenience script
│   └── tensorboard.sh  # script for launching TensorBoard
└── src/
    ├── cli.py         # CLI interface
    ├── data_prep.py   # data processing
    ├── model.py       # model architecture
    ├── losses/        # advanced loss functions
    │   ├── crf.py                # CRF implementation
    │   ├── focal_loss.py         # Focal Loss
    │   ├── label_smoothing.py    # Basic Label Smoothing
    │   ├── embedding_label_smoothing.py # Enhanced Label Smoothing
    │   ├── dice_loss.py          # Dice Loss
    │   └── compound_loss.py      # Loss Combination
    ├── lightning_module.py  # training logic
    ├── train.py       # training pipeline
    └── evaluate.py    # evaluation pipeline
```

---

## Data & Modeling Approach

1. **Data Preparation**  
   - The pipeline uses Hugging Face Datasets or your own custom dataset in `data/raw`.  
   - Tokenization and label alignment handle subwords carefully (important for Transformers).  
   - If domain-specific NER tasks are needed (e.g., addresses, place names, etc.), you can integrate domain knowledge, similar to **ArcGIS’s `EntityRecognizer`** approach, which has specialized training on geospatial text.

2. **Modeling**  
   - Choose any **Transformer** backbone (e.g., BERT, RoBERTa, Mistral, LLaMA).  
   - An optional **CRF layer** refines boundary predictions and ensures sequence-level consistency.  
   - For domain adaptation or smaller GPU constraints, you can use parameter-efficient fine-tuning methods (e.g., LoRA, Adapters).

3. **Training**  
   - Managed by **PyTorch Lightning** for clarity and reproducibility.  
   - Easy to incorporate advanced callbacks (e.g., early stopping, logging).  
   - Multiple loss functions (including CRF and specialized objectives) are integrated with a uniform interface, enabling quick experimentation.

4. **Evaluation**  
   - Computed with `seqeval` for standard metrics (precision, recall, F1).  
   - Simple interface to compare performance across multiple models, losses, and domain settings.

---

## Innovation: Advanced Loss Objectives

### Enhanced Label Smoothing with Embeddings

**Traditional label smoothing** uses a fixed probability (e.g., 0.1) for off-target labels. Our approach **learns label embeddings** to capture *relationships* between entity types:

1. **Label Embeddings**: Each label (e.g., `B-PER`, `I-PER`, `B-ORG`) is assigned a learnable vector.  
2. **Similarity-Based Softening**: The smoothing distribution is computed from similarity between the *gold label* embedding and all other label embeddings (e.g., `B-ORG` might get a higher "similarity score" than `B-LOC`).  
3. **Adaptive Weighting**: A temperature parameter (α) controls the distribution’s sharpness, balancing between hard targets and fully softened labels.

**Benefits**:  
- Preserves relationships between labels (makes “mistakes” less penalized when they are semantically close).  
- Improves generalization and domain transfer, inspired by a teacher-student strategy.

**Usage**:
```python
from src.losses.embedding_label_smoothing import EmbeddingLabelSmoothing

loss_fn = EmbeddingLabelSmoothing(
    num_labels=9,        # e.g., number of NER tags
    hidden_size=768,     # from your transformer model
    label_emb_dim=32,    # dimension for label embeddings
    smoothing_alpha=10.0 # temperature
)
```

---

## Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. CLI Entry Points

We provide a **Click-based CLI** for easy orchestration:

```bash
python -m src.cli prepare-data     # Preprocess and tokenize data
python -m src.cli run-train       # Train the model
python -m src.cli run-evaluate    # Evaluate on test or val split
```

### 2. Shell Script

Alternatively, use `scripts/run.sh`:

```bash
chmod +x scripts/run.sh
./scripts/run.sh all
```

- `prepare` for data prep  
- `train` for model training  
- `evaluate` for evaluation  

---

## Configuration

All main hyperparameters are in `config/config.yaml`:
```
model:
  base_model: "bert-base-cased"
  num_labels: 9
  loss:
    type: "embedding_label_smoothing"  # or "focal", "dice", "compound"
    params:
      label_emb_dim: 32
      smoothing_alpha: 10.0

training:
  lr: 5e-5
  epochs: 3
  batch_size: 8
  gpus: 1
  log_dir: "lightning_logs"  # Directory for storing logs
  checkpoint_dir: "checkpoints"  # Directory for storing model checkpoints
```
You can toggle between different losses by specifying `type`, and pass relevant parameters in `params`.

---

## References & Related Works

1. **ArcGIS `EntityRecognizer` Model**  
   - Provided insight into domain adaptation for specialized entity recognition (e.g., location-based entities).  
   - Showcases advanced fine-tuning and real-world usage scenarios.

2. **Hugging Face Transformers**  
   - For robust, pretrained language models and easy integration with token-classification tasks.

3. **PyTorch Lightning**  
   - For structured, reproducible experiments and flexible training loops.

4. **torchcrf** (or custom CRF)  
   - Improves entity boundary detection by modeling label dependencies.

5. Research in **label smoothing** & **soft targets**  
   - Inspired by teacher-student methods in knowledge distillation, which guided the label-embedding approach.

---

## License

This project is licensed under the **MIT License**.

---

**Happy experimenting!** If you have questions or suggestions, feel free to open an issue or contribute via pull requests.
