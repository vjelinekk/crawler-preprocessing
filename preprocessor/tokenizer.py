import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class TokenType(Enum):
    DATE = 1
    NUMBER = 2
    LINK = 3
    WORD = 4
    TAG = 5
    PUNCT = 6

@dataclass
class Token:
    processed_form: str
    position: int
    length: int
    token_type: TokenType = TokenType.WORD

    def __repr__(self):
        return self.processed_form



class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, document: str) -> list[Token]:
        raise NotImplementedError()

class SplitTokenizer(Tokenizer):
    def __init__(self, split_char: str):
        self.split_char = split_char

    def tokenize(self, document: str) -> list[Token]:
        tokens = []
        position = 0
        for word in document.split(self.split_char):
            token = Token(word, position, len(word))
            tokens.append(token)
            position += len(word) + 1
        return tokens

class RegexMatchTokenizer(Tokenizer):

    date_pattern = r'(\d{1,2}\.\s*\d{1,2}\.\s*\d{4}|\d{1,2}-\d{1,2}-\d{4}|\d{1,2}/\d{1,2}/\d{4})'  # matches dates like 1. 1. 2024, 1-1-2024, 1/1/2024
    num_pattern = r'(\d+[.,]?\d*)'  # matches numbers like 123, 123.123, 123,123
    link_pattern = r'(https?://\S+|www\.\S+|[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z][-a-zA-Z0-9]*(?:\.[a-zA-Z][-a-zA-Z0-9]*)*)'
    word_pattern = r'(\w+)'  # matches words
    html_tag_pattern = r'(<.*?>)'  # matches html tags
    punctuation_pattern = r'([^\w\s]+)'  # matches punctuation
    default_pattern = f'{date_pattern}|{num_pattern}|{link_pattern}|{word_pattern}|{html_tag_pattern}|{punctuation_pattern}'
    # default_pattern = r'(\d+[.,](\d+)?)|([\w]+)|(<.*?>)|([^\w\s]+)'

    def __init__(self, pattern: str=default_pattern):
        self.pattern = pattern

    def tokenize(self, document: str) -> list[Token]:
        tokens = []
        for match in re.finditer(re.compile(self.pattern, re.UNICODE), document):
            token = Token(match.group(), match.start(), match.end() - match.start(), TokenType(match.lastindex))
            tokens.append(token)
        return tokens

if __name__ == '__main__':
    document = "Hello, world! This is a test."
    document = 'příliš žluťoučký kůň úpěl ďábelské ódy. 20.25'
    tokenizer = SplitTokenizer(" ")
    tokens = tokenizer.tokenize(document)
    print(tokens)
    tokenizer = RegexMatchTokenizer()
    tokens = tokenizer.tokenize(document)
    print(tokens)