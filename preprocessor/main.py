import argparse
import sys
import config
import json
from typing import Iterable

from tokenizer import RegexMatchTokenizer, Tokenizer
from preprocess import PreprocessingPipeline
from collections import defaultdict, Counter
from pipeline_builder import PipelineBuilder


class Document:
    def __init__(self, text: str):
        self.text = text
        self.tokens = None
        self.vocab = None

    def tokenize(self, tokenizer: Tokenizer=None):
        tokenizer = tokenizer or RegexMatchTokenizer()
        self.tokens = tokenizer.tokenize(self.text)
        return self

    def preprocess(self, preprocessing_pipeline: PreprocessingPipeline):
        preprocessing_pipeline.preprocess(self.tokens, self.text)
        return self

def build_vocabulary(documents: Iterable[Document]):
    vocab = Counter()
    for doc in documents:
        vocab.update((token.processed_form for token in doc.tokens if token.processed_form != ''))
    return vocab

def write_weighted_vocab(vocab, file):
    for key, value in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
        file.write(f"{key} {value}\n")

def build_custom_pipeline():
    """Interactive function to build custom pipeline"""
    available_preprocessors = {
        '1': ('Lowercase', 'add_lowercase'),
        '2': ('Remove diacritics', 'add_remove_diacritics'),
        '3': ('Remove stopwords', 'add_remove_stopwords'),
        '4': ('Remove meaningless', 'add_remove_meaningless'),
        '5': ('Stemming', 'add_stemming'),
        '6': ('Lemmatization', 'add_lemmatization'),
    }
    
    print("\nBuild your own pipeline. Available preprocessors:")
    for key, (name, _) in available_preprocessors.items():
        print(f"{key}: {name}")
    print("Type 'build' when done")
    
    builder = PipelineBuilder()
    selections = []
    
    while True:
        choice = input("\nSelect preprocessor (or 'build' to finish): ")
        if choice.lower() == 'build':
            break

        if available_preprocessors[choice][0] in selections:
            print("Preprocessor already selected. Please try again.")
            continue
        
        if choice in available_preprocessors:
            name, method_name = available_preprocessors[choice]
            builder = getattr(builder, method_name)()
            selections.append(name)
            print(f"Added {name} to pipeline")
        else:
            print("Invalid choice. Please try again.")
    
    if not selections:
        print("No preprocessors selected. Using default stemming pipeline.")
        return PipelineBuilder.create_stemming_pipeline()
    
    print(f"\nBuilding pipeline with: {', '.join(selections)}")
    return builder.build()

def parse_arguments():
    description = """
    Text Preprocessing Pipeline
    """
    
    epilog = """
    EXAMPLES:
      python main.py stemming      # Use the stemming pipeline
      python main.py lemmatization # Use the lemmatization pipeline
      python main.py simple        # Use a simple pipeline without stemming/lemmatization
      python main.py custom        # Build your own pipeline interactively
    
    For more information, see README.md
    """
    
    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'pipeline_type',
        choices=['stemming', 'lemmatization', 'simple', 'custom'],
        help="Specify which preprocessing pipeline to use"
    )
    
    # If no arguments are provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    print(f"Loading documents from {config.DATA_PATH}")
    documents = []
    with open(config.DATA_PATH, 'r', encoding="utf-8") as f:
        documents = [Document(doc[config.JSON_KEY_TO_EXTRACT]).tokenize() for doc in json.load(f)]

    vocab = build_vocabulary(documents)
    print(f"Initial vocabulary size: {len(vocab)}")

    # Select pipeline based on command-line argument
    if args.pipeline_type == 'custom':
        pipeline = build_custom_pipeline()
    elif args.pipeline_type == 'stemming':
        print("Using stemming pipeline")
        pipeline = PipelineBuilder.create_stemming_pipeline()
    elif args.pipeline_type == 'lemmatization':
        print("Using lemmatization pipeline")
        pipeline = PipelineBuilder.create_lemmatization_pipeline()
    elif args.pipeline_type == 'simple':
        print("Using simple pipeline")
        pipeline = PipelineBuilder.create_simple_pipeline() 

    print("Preprocessing documents")
    documents = [doc.preprocess(pipeline) for doc in documents]

    vocab = build_vocabulary(documents)
    print(f"Final vocabulary size: {len(vocab)}")

    print(f"Saving vocabulary to {config.OUTPUT_PATH}")
    with open(config.OUTPUT_PATH, "w", encoding="utf-8") as f:
        write_weighted_vocab(vocab, f)