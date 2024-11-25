# TMN_DataGen

A tool for generating and processing dependency trees for training Graph Matching Networks on natural language inference and text similarity tasks.

## Overview

TMN_DataGen processes pairs of text sentences into dependency tree structures with associated features, ready for use in training graph matching networks. Key features:

- Multiple parser support (DiaParser, SpaCy) with flexible feature extraction
- Configurable text preprocessing and tokenization
- BERT-based word embeddings with caching
- Rich node features including POS tags, dependency relations, and morphological features 
- Support for batched processing and large datasets
- Built-in visualization tools
- Output format compatible with Graph Matching Networks

## Installation

1. Install the package:
```bash
pip install TMN_DataGen
```

2. Install required models:

### Parser Models

#### DiaParser Model
The default electra-base model will be downloaded automatically on first use.
- Provides dependency tree structure and relations
- Best accuracy for dependency parsing

#### SpaCy Models
Choose one of the following models (listed in order of accuracy/size):
```bash
# Transformer-based model (most accurate, requires GPU)
python -m spacy download en_core_web_trf

# Large model (good balance of accuracy/speed)
python -m spacy download en_core_web_lg

# Medium model
python -m spacy download en_core_web_md

# Small model (fastest, least accurate) 
python -m spacy download en_core_web_sm
```

Spacy provides:
- POS tagging
- Lemmatization
- Morphological features
- Named entity recognition (optional)

## Usage

### Basic Usage

```python
from TMN_DataGen import DatasetGenerator

# Initialize generator
generator = DatasetGenerator()

# Generate dataset
generator.generate_dataset(
    sentence_pairs=[("The cat chases the mouse.", "A mouse is being chased.")],
    labels=["entailment"],
    output_path="output.json"
)
```

### Processing Large Datasets

For large datasets like SNLI, use the included batch processing script:

```bash
python run.py --input_file snli_data.jsonl --out_dir processed_data/
```

### Configuration

The package uses a hierarchical configuration system with reasonable defaults that can be overridden:

1. Parser Configuration
- Parser selection and settings
- Feature source mapping
- Batch sizes

2. Preprocessing Configuration  
- Strictness levels (0-3)
- Tokenization options
- Text normalization settings

3. Feature Configuration
- Word embedding model selection
- Feature dimensionality
- Caching settings

Example configuration override:
```python
parser_config = {
    'parser': {
        'type': 'multi',
        'parsers': {
            'diaparser': {'enabled': True},
            'spacy': {'enabled': True}
        }
    }
}

generator.generate_dataset(
    sentence_pairs=pairs,
    labels=labels,
    output_path="output.json",
    parser_config=parser_config,
    verbosity='normal'
)
```

## Output Format

The generated dataset is saved in JSON format with the following structure:
```json
{
    "graph_pairs": [
        [{
            "node_features": [...],
            "edge_features": [...],
            "from_idx": [...],
            "to_idx": [...],
            "graph_idx": [...],
            "n_graphs": 1
        },
        {
            // Second graph in pair
        }],
        // Additional pairs...
    ],
    "labels": [1, -1, 0, ...]  // 1: entailment, -1: contradiction, 0: neutral
}
```

## Feature Details

Node features include:
- BERT word embeddings (768-dim)
- POS tag one-hot encoding
- Morphological feature vectors
- Dependency relation type embedding

Edge features encode:
- Dependency relation types
- Structural information

## Requirements
See requirements.txt for full list. Key dependencies:
```
torch>=2.0.0
diaparser>=1.1.3
spacy>=3.0.0
transformers>=4.30.0
```

Optional dependencies:
```
stanza>=1.2.3  # For enhanced tokenization
```

## Notes
- The package caches word embeddings by default to improve performance
- GPU support is enabled automatically when available
- Processes both English and multilingual text (with appropriate models)
