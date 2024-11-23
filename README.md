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

## Configuration

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
