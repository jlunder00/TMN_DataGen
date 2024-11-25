# TMN_DataGen

A tool for generating and processing dependency trees for training Graph Matching Networks on Trees.

## Installation

1. Install the package:
```bash
pip install TMN_DataGen
```

2. Install required models:

### SpaCy Models
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

### DiaParser Models
The default electra-base model will be downloaded automatically, but other options are available.
NOTE: Diaparser only gives a dependency parse and dependency roles (usually the best at this), but it does not give lemmas, part of speech, or other text features. Spacy and other models do give this. Included is the multi model configuration to select which features (tree structure, part of speech, lemma, etc.) come from which of multiple models.

### Model Configuration

Models can be specified in your configuration file:
```yaml
parser:
  type: multi
  batch_size: 32
  parsers:
    diaparser:
      enabled: true
      model_name: "en_ewt.electra-base"  # or other DiaParser model
    spacy:
      enabled: true
      model_name: "en_core_web_trf"  # or any installed SpaCy model
```


## Text Preprocessing and Tokenization

TMN_DataGen provides configurable text preprocessing and tokenization options:

### Preprocessing Levels

The preprocessing pipeline has 4 levels of strictness:

- **Level 0 (None)**: No preprocessing
- **Level 1 (Basic)**: Whitespace normalization and punctuation handling
- **Level 2 (Medium)**: Unicode normalization and non-ASCII removal
- **Level 3 (Strict)**: Case normalization and accent removal

### Tokenization Options

Two tokenization approaches are available:

1. **Regex Tokenizer**: Simple rule-based tokenization using word boundaries
2. **Stanza Tokenizer**: Neural tokenizer with better handling of complex cases

### Configuration

Example configuration in yaml:

```yaml
preprocessing:
  strictness_level: 2  # 0-3
  tokenizer: "regex"   # "regex" or "stanza"
  language: "en"
  preserve_case: false
  remove_punctuation: true
  normalize_unicode: true
  remove_numbers: false
  max_token_length: 50
  min_token_length: 1
```

### Example Usage

```python
from TMN_DataGen import DiaParserTreeParser
from omegaconf import OmegaConf

# Load config with strict preprocessing
config = OmegaConf.load('configs/preprocessing_strict.yaml')
parser = DiaParserTreeParser(config)

# Parse with preprocessing
sentence = "The café is nice!"
tree = parser.parse_single(sentence)
# Result: Normalized and tokenized sentence processed into dependency tree
```

for languages that require specialized tokenization, the Stanza tokenizer is recommended:
```yaml
preprocessing:
  strictness_level: 1
  tokenizer: "stanza"
  language: "en"
```

Note: Stanza tokenizer requires additional dependencies. Install with:;
```bash
pip install TMN_DataGen[stanza]
```


