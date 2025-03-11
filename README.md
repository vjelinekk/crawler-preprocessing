# Text Preprocessing Library for Information Retrieval

## Features

- **Tokenization** - Extract tokens from text documents
- **Modular preprocessing** - Use various preprocessing techniques:
  - Lowercase conversion
  - Diacritics removal
  - Stopwords removal
  - Meaningless tokens removal
  - Stemming (using external stemmer)
  - Lemmatization (using UDPipe)
- **Pipeline architecture** - Chain preprocessors together
- **Builder pattern** - Fluent interface for creating processing pipelines
- **Vocabulary building** - Generate and save vocabulary with term frequencies

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install directly
pip install ufal.udpipe
```

## Usage

### Basic Usage

```python
from tokenizer import RegexMatchTokenizer
from preprocess import PreprocessingPipeline
from pipeline_builder import PipelineBuilder

# Create a document
document = Document("Toto je ukázkový český text.")

# Tokenize
tokenizer = RegexMatchTokenizer()
document.tokenize(tokenizer)

# Process with a predefined pipeline
pipeline = PipelineBuilder.create_lemmatization_pipeline()
document.preprocess(pipeline)
```

### Creating Custom Pipelines

```python
# Using the builder pattern
pipeline = PipelineBuilder() \
    .add_lowercase() \
    .add_remove_diacritics() \
    .add_remove_stopwords() \
    .add_lemmatization() \
    .build()

# Process documents
documents = [doc.preprocess(pipeline) for doc in documents]

# Build vocabulary
vocab = build_vocabulary(documents)
```

### Executing the code

- You can execute your code with:
```bash
python main.py
```

## Project Structure

```
preprocessing/
├── preprocessor/
│   ├── stopwords/
│   │   └── stopwords_czech.txt  # Czech stopwords list
│   ├── udpipe/
│   │   └── czech-pdt-ud-2.5-191206.udpipe  # UDPipe model
│   ├── stemmer/
│   │   └── stemmer          # External stemmer executable
│   ├── main.py              # Example usage script
│   ├── tokenizer.py         # Tokenization logic
│   ├── preprocess.py        # Preprocessing components
│   ├── pipeline_builder.py  # Pipeline builder implementation
├── data/
│   ├── movies.json          # Sample input data
│   └── vocab.txt            # Generated vocabulary
```

## Components

### Tokenizer

Extracts tokens from text documents.

### TokenPreprocessor

Abstract base class for all preprocessing operations.

### Available Preprocessors

- **LowercasePreprocessor** - Convert text to lowercase
- **RemoveDiacriticsPreprocessor** - Remove diacritics (accents) from text
- **RemoveStopwordsPreprocessor** - Filter out common stopwords
- **RemoveMeaninglessPreprocessor** - Remove single characters and punctuation
- **StemmingPreprocessor** - Reduce words to their stems
- **LemmatizationPreprocessor** - Reduce words to their base dictionary form

### PipelineBuilder

Provides a fluent interface for creating preprocessing pipelines.

## Dependencies

- Python 3.6+
- ufal.udpipe (for lemmatization)
- External stemmer executable (for stemming)
    - If you don not use MacOS with M1 CPU you need to build the stemmer executable yourself:
    ```bash
    cd preprocessor/stemmer
    make
    ```
    - Also if you are on Windows machine you need to change the constant `STEMMER_EXECUTABLE_PATH` in `config.py` from `'./stemmer/stemmer'` to `'./stemmer/stemmer.exe'`

## License

[MIT License](LICENSE)