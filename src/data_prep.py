import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
import os

def prepare_data(config_path: str, ratio: float = 1.0):
    """
    Prepare and process the 'eriktks/conll2003' dataset for a token-level
    classification task (NER). The processed splits (train/val/test) are saved
    to disk in a 'data/processed/' directory.
    Additionally, saves the label list for NER tags to be used during evaluation.
    
    Args:
        config_path: Path to the configuration file.
        ratio: Fraction of the dataset to retain (default is 1.0, meaning no reduction).
    """

    # 1. Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Extract relevant config values
    dataset_name = config['data']['dataset_name']  # e.g., "eriktks/conll2003"
    max_length = config['data']['max_length']      # e.g., 128
    base_model = config['model']['base_model']     # e.g., "openlm-research/llama-3.2-3b"

    save_path = "data/processed"
    os.makedirs(save_path, exist_ok=True)
    # 3. Load dataset
    raw_datasets = load_dataset(dataset_name)
    # raw_datasets should now contain keys: ["train", "validation", "test"]

    # Save label list for NER tags
    label_list = raw_datasets['train'].features['ner_tags'].feature.names
    with open(os.path.join(save_path, 'label_list.txt'), 'w') as label_file:
        for label in label_list:
            label_file.write(label + "\n")

    # Reduce dataset size according to the ratio
    if ratio < 1.0:
        for split in ['train', 'validation', 'test']:
            raw_datasets[split] = raw_datasets[split].shuffle(seed=42).select(range(int(len(raw_datasets[split]) * ratio)))

    # 4. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    # 5. Preprocessing function:
    #    - Applies word-level to subword-level label alignment
    #    - Uses -100 as the "ignore index" for subword tokens
    def tokenize_and_align_labels(examples):
        # Tokenize
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
            padding="max_length"  # Optional: ensures fixed length
        )

        aligned_labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens (CLS, SEP, PAD, etc.)
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # Take the label for the *first* subword only
                    label_ids.append(ner_tags[word_idx])
                else:
                    # Mark subsequent subwords of the same token as -100
                    label_ids.append(-100)
                previous_word_idx = word_idx

            aligned_labels.append(label_ids)

        tokenized_inputs["labels"] = aligned_labels
        return tokenized_inputs

    # 6. Map the preprocessing function across splits
    #    remove_columns drops original columns (tokens, pos_tags, etc.) to keep only model inputs
    processed_train = raw_datasets["train"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )
    processed_val = raw_datasets["validation"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )
    processed_test = raw_datasets["test"].map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
    )

    # 7. Save processed datasets
    processed_train.save_to_disk(os.path.join(save_path, "train"))
    processed_val.save_to_disk(os.path.join(save_path, "val"))
    processed_test.save_to_disk(os.path.join(save_path, "test"))

    # Save dataset metadata
    def save_dataset_metadata(dataset, dataset_name: str, output_dir: str = 'data'):
        """
        Save dataset metadata to a YAML file.
        
        Args:
            dataset: HuggingFace DatasetDict
            dataset_name: Name of the dataset from config
            output_dir: Directory to save metadata
        """
        import os
        import yaml
        from datetime import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract metadata
        metadata = {
            'dataset_info': {
                'name': dataset_name,  # Use the name from config
                'split_sizes': {
                    'train': len(dataset['train']),
                    'validation': len(dataset['validation']),
                    'test': len(dataset['test'])
                },
                'timestamp': datetime.now().isoformat()
            },
            'label_info': {
                'ner_tags': {
                    'num_classes': len(dataset['train'].features['ner_tags'].feature.names),
                    'names': dataset['train'].features['ner_tags'].feature.names
                },
                'pos_tags': {
                    'num_classes': len(dataset['train'].features['pos_tags'].feature.names),
                    'names': dataset['train'].features['pos_tags'].feature.names
                },
                'chunk_tags': {
                    'num_classes': len(dataset['train'].features['chunk_tags'].feature.names),
                    'names': dataset['train'].features['chunk_tags'].feature.names
                }
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(output_dir, 'dataset_metadata.yaml')
        with open(metadata_path, 'w') as f:
            yaml.safe_dump(metadata, f, default_flow_style=False)
        
        return metadata_path

    metadata_path = save_dataset_metadata(raw_datasets, dataset_name, 'data')
    
    print("Data preparation complete! Processed datasets saved to disk.")
