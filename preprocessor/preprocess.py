from abc import ABC, abstractmethod
from string import punctuation
from unicodedata import category, normalize
from tokenizer import Token 
from subprocess import run
from ufal.udpipe import Model, Sentence

STOPWORDS_PATH = './stopwords/stopwords_czech.txt'
UDPIDE_MODEL_PATH = './udpipe/czech-pdt-ud-2.5-191206.udpipe'
STEMMER_EXECUTABLE_PATH = './stemmer/stemmer'


class TokenPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, token: Token, document: str) -> Token:
        raise NotImplementedError()

    def preprocess_all(self, tokens: list[Token], document: str) -> list[Token]:
        return [self.preprocess(token, document) for token in tokens]


class LowercasePreprocessor(TokenPreprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        if token.processed_form == '':
            return token

        token.processed_form = token.processed_form.lower()
        return token


class RemoveDiacriticsPreprocessor(TokenPreprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        if token.processed_form == '':
            return token

        token.processed_form = ''.join(
            c for c in normalize('NFD', token.processed_form)
            if category(c) != 'Mn'
        )

        return token
    

class RemoveStopwordsPreprocessor(TokenPreprocessor):
    def __init__(self):
        self.stopwords = set()
        with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
            self.stopwords = set(f.read().splitlines())

    def preprocess(self, token: Token, document: str) -> Token:
        if token.processed_form == '':
            return token

        if token.processed_form in self.stopwords:
            token.processed_form = ''
        
        return token


class RemoveMeaninglessPreprocessor(TokenPreprocessor):
    def preprocess(self, token: Token, document: str) -> Token:
        if token.processed_form == '':
            return token

        if len(token.processed_form) <= 1 or all(char in punctuation for char in token.processed_form):
            token.processed_form = ''
        
        return token


class StemmingPreprocessor(TokenPreprocessor):
    def __init__(self, stemmer_executable_path: str = STEMMER_EXECUTABLE_PATH):
        self.stemmer_executable_path = stemmer_executable_path

    def preprocess(self, token: Token, document: str) -> Token:
        if token.processed_form == '':
            return token
        
        result = run(
            [self.stemmer_executable_path],
            input=token.processed_form,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            token.processed_form = result.stdout.strip()
        else:
            print(f"Stemming failed for token: {token.processed_form}")

        return token


class LemmatizationPreprocessor(TokenPreprocessor):
    def __init__(self, model_path: str = UDPIDE_MODEL_PATH):
        self.model = Model.load(model_path)
        if not self.model:
            raise Exception(f"Model {model_path} not found")

    def preprocess(self, token: Token, document: str) -> Token:
        if token.processed_form == '':
            return token

        sentence = Sentence()
        word = sentence.addWord(token.processed_form)

        self.model.tag(sentence, Model.DEFAULT)

        token.processed_form = word.lemma

        return token
        


class PreprocessingPipeline:
    def __init__(self, preprocessors: list[TokenPreprocessor]):
        self.preprocessors = preprocessors

    def preprocess(self, tokens: list[Token], document: str) -> list[Token]:
        for preprocessor in self.preprocessors:
            tokens = preprocessor.preprocess_all(tokens, document)
        return tokens