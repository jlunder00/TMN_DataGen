[//]: # (Authored by: Jason Lunder EWUID: 01032294)

# TMN_DataGen

A tool for generating and processing dependency trees for training Graph Matching Networks on natural language inference and text similarity tasks.

Note: this has not been tested in other environments. It is possible there are issues preventing it from working in environments other than my own.    

### Creating Embedding Cache for Demo or Evaluation

The embedding cache is required for all usage scenarios. Here's how to create it:

1. **[Follow package setup instructions here](#installation)**

1. **Generate a small dataset**: Processing even a small dataset will create the necessary embedding cache structure.

```bash
# Process small SNLI test data to create minimal cache
python -m TMN_DataGen.run process \
  --input_path data/snli_1.0/snli_1.0_test.jsonl \
  --out_dir processed_data/test \
  --dataset_type snli \
  --num_workers 4 \
  --num_partitions 10  # number of partitions
  --batch_size 10
  --spacy_model en_core_web_sm \
  --max_lines 100  # Process only 100 lines
 ```
TODO: explain concurrency system. Defaults to not being used so should be fine.

2. **Configure cache path**: Ensure the embedding cache path is properly set.

```yaml
# In your configuration
feature_extraction:
  embedding_cache_dir: "embedding_cache"  # Path to created cache
  do_not_store_word_embeddings: true      # Essential for runtime embedding retrieval
```

3. **Next steps**:
   - For running the demo script: See `Tree-Matching-Networks/scripts/README.md`
   - For evaluation with test data: See `Tree-Matching-Networks/Tree_Matching_Networks/LinguisticTrees/README.md`
   - For training with larger datasets: See the training section in the LinguisticTrees README

### Important Notes

- You only need to generate a full dataset if you plan to train or evaluate
- The embedding cache is required for all scenarios (demo, evaluation, training)
- Always use `do_not_store_word_embeddings: true` in the configuration
- For datasets with poor word boundaries like PatentMatch, use `strictness_level: 2`

## Overview

TMN_DataGen processes text data into dependency tree structures with associated features, ready for use in training Graph Matching Networks with contrastive learning approaches. It supports multiple input formats and can generate tree representations for various NLP tasks.

## Features

- **Multiple Parser Support**: DiaParser and SpaCy with configurable feature extraction
- **Flexible Input Formats**: Process sentence pairs or text groups from various datasets
- **Contrastive Learning Support**: Format data for InfoNCE and other contrastive losses
- **Configurable Text Preprocessing**: Multiple strictness levels and tokenization options
- **Word Embeddings**: BERT-based word embeddings with efficient caching
- **Rich Feature Extraction**: POS tags, dependency relations, and morphological features
- **Efficient Processing**: Batched processing with multiprocessing support
- **Visualization Tools**: Built-in tree visualization utilities
- **Output Format**: Compatible with Graph/Tree Matching Networks

## Installation

Install from source:
```bash
git clone https://github.com/jlunder00/TMN_DataGen.git
cd TMN_DataGen
pip install -e .
```

Note: Currently, for use with Tree-Matching-Networks, the package MUST be installed in editable mode (-e) and in the same directory as Tree-Matching-Networks.

## Required Dependencies

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

### Word Embeddings and Vocabulary

For tokenization and embedding generation, the system uses:

1. **Word2Vec Vocabulary**: Used for tokenization and fixing word boundaries
   - Download from: [Google News Vectors](https://www.kaggle.com/datasets/adarshsng/googlenewsvectors)
   - Set path in the preprocessing config (default: `vocab_model_path`)
   
   ```yaml
   preprocessing:
     vocab_model_path: "/path/to/GoogleNews-vectors-negative300.bin"
     vocab_limit: 500000  # Limit to top N common words
   ```

2. **BERT Embeddings**: Generated automatically for text tokens
   - Cached to disk for efficiency
   - Configure cache location in feature extraction config

## Supported Datasets

TMN_DataGen supports various datasets. Here are the main ones used and tested:

### SNLI (Stanford Natural Language Inference)
- **Description**: Sentence pairs with entailment labels (entailment, contradiction, neutral)
- **Website**: [https://nlp.stanford.edu/projects/snli/](https://nlp.stanford.edu/projects/snli/)
- **Direct Download**: [https://nlp.stanford.edu/projects/snli/snli_1.0.zip](https://nlp.stanford.edu/projects/snli/snli_1.0.zip)
    - Includes test/train/val splits (Fine tuning task for CSCD 584 project)
- **Format**: JSONL
- **Dataset Type**: `snli`
- **Status**: Fully tested and supported

### WikiQS (Wiki Question Similarity)
- **Description**: Sets of similar questions from Wikipedia
- **Website**: [https://github.com/afader/oqa#wikianswers-corpus](https://github.com/afader/oqa#wikianswers-corpus)
- **Download Site**: [https://knowitall.cs.washington.edu/oqa/data/wikianswers/](https://knowitall.cs.washington.edu/oqa/data/wikianswers/)
- **Format**: Tab-separated text files with question groups
- **Dataset Type**: `wiki_qs`
- **Status**: Tested with single partition as input, *might* support multiple partitions

### SemEval
- **Description**: Semantic textual similarity data from SemEval competitions
- **Download Site**: [https://github.com/brmson/dataset-sts/tree/master/data/sts/semeval-sts/all](https://github.com/brmson/dataset-sts/tree/master/data/sts/semeval-sts/all)
    - Used 2015.train.tsv, 2015.test.tsv, and 2015.val.tsv (separate fine tuning task)
- **Format**: JSONL
- **Dataset Type**: `semeval`
- **Status**: Supported

### AmazonQA
- **Description**: Question-answer pairs from Amazon product Q&A
- **Download Site**: [https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/qa/](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/qa/)
    - Used all available, both multi answer and single answer (pretraining data) 
- **Format**: Gzipped JSON
- **Dataset Type**: `amazon_qa`
- **Status**: Supported

### PatentMatch
- **Description**: Patent claims and prior art reference pairs
- **Download Site**: [https://my.hidrive.com/share/rwfam92omy](https://my.hidrive.com/share/rwfam92omy)
    - Used patentmatch_test_balanced.zip and patentmatch_train_balanced.zip (separate finetuning task)
- **Format**: TSV
- **Dataset Type**: `patentmatch`
- **Status**: Supported

### ParaNMT-50M
- **Description**: Large dataset of paraphrased sentence pairs (50 million)
- **Download**: [PLACEHOLDER: Link to ParaNMT-50M data]
    - Did not use for this project
- **Note**: Support implemented but not extensively tested
- **Status**: Experimental support

## Configuration

TMN_DataGen uses a hierarchical configuration system with YAML files. Default configurations are included in the package, but you can override them. Each configuration section controls different aspects of the processing pipeline:

### Parser Configuration
```yaml
parser:
  type: "multi"               # Parser type - "multi" uses both parsers for best results
  batch_size: 32              # Batch size for parser processing
  min_tokens: 3               # Skip sentences with fewer than this many tokens
  max_tokens: 100             # Skip sentences with more than this many tokens
  min_nodes: 3                # Skip trees with fewer than this many nodes
  max_nodes: 325              # Skip trees with more than this many nodes
  parsers:
    diaparser:                # DiaParser configuration
      enabled: true           # Whether to use this parser
      model_name: "en_ewt.electra-base"  # DiaParser model name
    spacy:                    # SpaCy configuration
      enabled: true           # Whether to use this parser
      model_name: "en_core_web_sm"  # SpaCy model name
  feature_sources:            # Which parser to use for which features
    tree_structure: "diaparser"  # DiaParser gives best tree structure
    pos_tags: "spacy"            # SpaCy is better for POS tagging
    lemmas: "spacy"              # SpaCy provides lemmatization
    dependency_labels: "diaparser"  # DiaParser has better dep labels
    morph_features: "spacy"      # SpaCy provides morphological features
```

### Preprocessing Configuration
```yaml
preprocessing:
  strictness_level: 2         # Preprocessing strictness: 
                             # 0=None (raw text)
                             # 1=Basic (whitespace/punct handling)
                             # 2=Medium (Unicode normalization)
                             # 3=Strict (case normalization)
  tokenizer: "regex"         # Tokenizer to use: "regex" (faster) or "stanza" (more accurate)
  language: "en"             # Language code
  preserve_case: false       # Whether to preserve case
  remove_punctuation: true   # Whether to remove punctuation
  normalize_unicode: true    # Normalize Unicode characters
  vocab_model_path: "/path/to/GoogleNews-vectors-negative300.bin"  # Word2Vec vocab 
  vocab_limit: 500000        # Limit vocab to top N words (for memory reasons)
```

### Feature Configuration
```yaml
feature_extraction:
  word_embedding_model: "bert-base-uncased"  # Model for word embeddings
  cache_embeddings: true      # Cache embeddings to disk
  embedding_cache_dir: "embedding_cache"  # Where to store cache
  use_gpu: true              # Use GPU for embedding generation
  do_not_store_word_embeddings: true  # Store embeddings in cache, not output
                                     # (reduces output file size dramatically)
```

### Output Format Configuration
```yaml
output_format:
  type: "infonce"  # Output in grouped format
  paired: true     # Whether text pairs are paired (true) or single sets of texts (false). Ex: wikiqs is not paired, SNLI is.
  normalize: null  # Optional normalization for continuous scores
  label_map:       # Map string labels to numeric values
    "entailment": 1
    "neutral": 0
    "contradiction": -1
```

Note: "InfoNCE" is currently the only format supported and is somewhat misleadingly named. Despite the name, this format is compatible with most loss types including direct labels, triplet loss, and true InfoNCE loss. The format simply organizes text trees into logical groups rather than flat pairs, making it more flexible for different training approaches. It will be renamed in a future version.

## Usage

### Command Line Interface

For large datasets, use the included batch processing script:

```bash
python /path/to/TMN_DataGen/run.py process \
  --input_path data/snli_1.0/snli_1.0_dev.jsonl \
  --out_dir processed_data/dev \
  --dataset_type snli \
  --spacy_model en_core_web_sm \
  --verbosity normal \
  --batch_size 100 \
  --num_partitions 2
```

#### Command Line Arguments

The key command line arguments are:

- `--input_path` / `-ip`: Path to input file or directory
- `--out_dir` / `-od`: Output directory where processed data will be saved
- `--dataset_type` / `-dt`: Type of dataset to process (`snli`, `wiki_qs`, `amazon_qa`, `patentmatch`, `semeval`)
- `--spacy_model` / `-sm`: SpaCy model to use (e.g., `en_core_web_sm`, `en_core_web_lg`)
- `--verbosity` / `-v`: Logging verbosity level (`quiet`, `normal`, `debug`)
- `--batch_size` / `-bs`: Number of text pairs per batch
- `--max_lines` / `-ml`: Maximum number of lines to process (for testing)
- `--num_partitions` / `-n`: Split output into this many partitions
- `--workers` / `-w`: Number of worker processes for parallel operations

### Data Splitting and Processing

TMN_DataGen focuses solely on converting text data to tree format and does not handle train/validation/test splits. This is by design to allow flexibility in how users define their splits.

### Creating Dataset Splits

- **Pre-split datasets**: For datasets like SNLI that come with predefined splits, simply process each split file separately.
- **Custom splitting**: For datasets without predefined splits, first create your train/val/test splits using external tools, then process each resulting file.

For convenience, some basic scripts for splitting datasets can be found in the `TMN_DataGen/scripts` directory, such as:
- `split_wikiqs.py` - Creates train/val/test splits for WikiQS data

Additional dataset splitting functionality will be added in future versions.

For each dataset type, use the appropriate `--dataset_type` parameter:

```bash
# SNLI dataset
python -m TMN_DataGen.run process \
  --input_path data/snli_1.0/snli_1.0_dev.jsonl \
  --out_dir processed_data/snli_dev \
  --dataset_type snli

# Wiki Question Similarity
python -m TMN_DataGen.run process \
  --input_path data/wikiquestions/qs.txt \
  --out_dir processed_data/wiki_qs \
  --dataset_type wiki_qs

# Amazon QA data
python -m TMN_DataGen.run process \
  --input_path data/amazon_qa/qa_data.json.gz \
  --out_dir processed_data/amazon_qa \
  --dataset_type amazon_qa

# Patent match data
python -m TMN_DataGen.run process \
  --input_path data/patentmatch/patent_pairs.tsv \
  --out_dir processed_data/patents \
  --dataset_type patentmatch
```

### Dataset-Specific Configurations

Each dataset type has different characteristics that may benefit from specific configurations:

#### SNLI Dataset
- Contains premise-hypothesis pairs with labeled entailment relationships
- Default configurations work well for this dataset
- Recommended configuration:
```yaml
output_format:
  type: "infonce"
  paired: true
  label_map:
    "entailment": 1
    "neutral": 0
    "contradiction": -1
```

#### WikiQS (Wiki Question Similarity)
- Contains groups of related questions
- No explicit labels, all questions in a group are considered similar
- Recommended configuration:
```yaml
preprocessing:
  strictness_level: 2  # Needed for vocab filtering
output_format:
  type: "infonce"
  paired: true  # Each group creates pairs of questions
```

#### AmazonQA
- Contains question-answer pairs
- Recommended configuration:
```yaml
preprocessing:
  strictness_level: 2  # Good for vocabulary filtering
output_format:
  type: "infonce"
  paired: true
```

#### PatentMatch
- Contains technical language with specialized terminology
- **Requires strictness level 2** for vocab filtering and deep word splitting
- Recommended configuration:
```yaml
preprocessing:
  strictness_level: 2  # Critical for messy text
output_format:
  type: "infonce"
  paired: true
  label_map:
    "1": 1    # Match
    "0": -1   # Non-match
```

### Merging Partitioned Output

After processing large datasets into partitions, you can merge them:

```bash
python -m TMN_DataGen.run merge \
  --input_files processed_data/dev/part_0.json processed_data/dev/part_1.json \
  --output processed_data/dev/final_dataset.json
```

### Programmatic API

**Note**: The programmatic API is experimental and still under development. Currently, most functionality is exposed through the command-line interface shown above. See the demo script in Tree-Matching-Networks for examples of how to use TMN_DataGen components programmatically.

## Integration with Tree Matching Networks

TMN_DataGen is designed to be used with the Tree Matching Networks package. A demo script is available in the Tree-Matching-Networks repository showing how to:

1. Preprocess text pairs
2. Parse them into dependency trees
3. Convert to the appropriate format
4. Run inference with a trained model

## Embedding Cache

Word embeddings are cached to improve performance. The cache is automatically created when you first run the code and will grow as new words are encountered.

**Important**: The embedding cache can become quite large (several GB) for large datasets. You can configure its location in the `feature_extraction` config section:

```yaml
feature_extraction:
  embedding_cache_dir: "/path/to/embedding_cache"
```

The cache uses a sharded storage system to avoid memory issues with large vocabularies.

## Output Format

The generated dataset is saved in JSON format. When using the grouped format (currently labeled "infonce"), the structure is:

```json
{
  "version": "1.0",
  "format": "infonce",
  "requires_word_embeddings": true,  // Whether word embeddings need to be loaded at runtime
  "groups": [
    {
      "group_id": "...",  // Unique identifier for the group
      "text": "The cat chases the mouse.",  // Original text A
      "trees": [  // List of trees parsed from text A (may be multiple sentences)
        {
          "node_features": [...],  // Node feature vectors
          "edge_features": [...],  // Edge feature vectors
          "from_idx": [...],  // Source node indices for edges
          "to_idx": [...],    // Target node indices for edges
          "graph_idx": [...], // Graph indices (all 0 for single graph)
          "n_graphs": 1,      // Number of graphs
          "node_texts": [...], // Original words
          "node_features_need_word_embs_prepended": true, // If embeddings need to be added
          "text": "The cat chases the mouse." // Original sentence
        }
      ],
      "text_b": "A mouse is being chased.",  // Original text B (if paired=true)
      "trees_b": [...],  // List of trees from text B (if paired=true)
      "label": "entailment"  // Label for the group
    },
    ...
  ]
}
```

Each tree represents a sentence's dependency structure and contains:

- **node_features**: Feature vectors for each node (word) in the tree - when not storing word embeddings, this is the one hot encoding for part of speech and other morphological features. Word embeddings prepended if stored or at runtime if not stored.
- **edge_features**: Feature vectors for each edge (one hot encoding of dependency relation)
- **from_idx/to_idx**: Indices defining the tree structure
- **node_texts**: Original word/lemma pairs for embedding lookup
- **node_features_need_word_embs_prepended**: Flag indicating whether embeddings need to be loaded from cache at runtime (when `do_not_store_word_embeddings` is true)

This format is designed to:
1. Group related trees together (from one or more original texts)
2. Preserve original text for reference
3. Efficiently handle word embeddings through caching
4. Support both paired and unpaired datasets

## Known Issues and Limitations

1. **Large Memory Usage**: Processing large datasets can require significant memory. Use batching and partitioning to manage memory usage.
2. **Path Configuration**: Some components have filepath configurations that must be set correctly:
   - Word2Vec model path in preprocessing config
   - Embedding cache directory in feature extraction config
3. **GPU Requirements**: For optimal performance, a GPU is recommended, especially for transformer-based models
4. **Experimental API**: The programmatic API is still experimental; command-line usage is more stable
5. **Large Output Files**: Output files can be very large even with embedding caching; use partitioning for large datasets

## Requirements

See requirements.txt for full list. Key dependencies:
```
torch>=2.0.0
numpy>=1.20.0
diaparser>=1.1.3
spacy>=3.0.0
transformers>=4.30.0
tqdm>=4.65.0
omegaconf>=2.3.0
graphviz
stanza>=1.2.3 # Optional, for enhanced tokenization
boto3
multiprocess
pyarrow
pandas
polars
scipy
scikit-learn
python-dotenv
gensim # For word2vec loading
english_words # For tokenization improvements
```

## License

MIT
