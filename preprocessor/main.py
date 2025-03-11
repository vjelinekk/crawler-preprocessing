import json
from typing import Iterable

from tokenizer import RegexMatchTokenizer, Tokenizer
from preprocess import PreprocessingPipeline
from collections import defaultdict, Counter
from pipeline_builder import PipelineBuilder

DATA_PATH = "../data/movies.json"
JSON_KEY_TO_EXTRACT = "description"
OUTPUT_PATH = "../data/vocab.txt"

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

if __name__ == '__main__':
    documents = []
    with open(DATA_PATH, 'r', encoding="utf-8") as f:
        documents = [Document(doc[JSON_KEY_TO_EXTRACT]).tokenize() for doc in json.load(f)]

    vocab = build_vocabulary(documents)
    print(len(vocab))

    pipeline = PipelineBuilder.create_stemming_pipeline()
    # pipeline = PipelineBuilder.create_lemmatization_pipeline()
    # pipeline = PipelineBuilder.create_simple_pipeline()
    documents = [doc.preprocess(pipeline) for doc in documents]

    vocab = build_vocabulary(documents)
    print(len(vocab))

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        write_weighted_vocab(vocab, f)