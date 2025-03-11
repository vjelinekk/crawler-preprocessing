from preprocess import (
    TokenPreprocessor,
    PreprocessingPipeline,
    LowercasePreprocessor,
    RemoveStopwordsPreprocessor,
    RemoveMeaninglessPreprocessor,
    RemoveDiacriticsPreprocessor,
    StemmingPreprocessor,
    LemmatizationPreprocessor,
)

import config


class PipelineBuilder:
    """
    Builder class for creating preprocessing pipelines. 
    """ 
    def __init__(self):
        self.preprocessors: PreprocessingPipeline = []

    def add_lowercase(self) -> 'PipelineBuilder':
        self.preprocessors.append(LowercasePreprocessor())
        return self

    def add_remove_stopwords(self) -> 'PipelineBuilder':
        self.preprocessors.append(RemoveStopwordsPreprocessor())
        return self
    
    def add_remove_meaningless(self) -> 'PipelineBuilder':
        self.preprocessors.append(RemoveMeaninglessPreprocessor())
        return self
    
    def add_remove_diacritics(self) -> 'PipelineBuilder':
        self.preprocessors.append(RemoveDiacriticsPreprocessor())
        return self
    
    def add_stemming(self, stemmer_path: str = config.STEMMER_EXECUTABLE_PATH) -> 'PipelineBuilder':
        self.preprocessors.append(StemmingPreprocessor(stemmer_path))
        return self
    
    def add_lemmatization(self, model_path: str = config.UDPIDE_MODEL_PATH) -> 'PipelineBuilder':
        self.preprocessors.append(LemmatizationPreprocessor(model_path))
        return self
    
    def add_custom(self, preprocessor: TokenPreprocessor) -> 'PipelineBuilder':
        self.preprocessors.append(preprocessor)
        return self

    def build(self) -> PreprocessingPipeline:
        return PreprocessingPipeline(self.preprocessors)

    @staticmethod
    def create_stemming_pipeline(stemmer_path: str = config.STEMMER_EXECUTABLE_PATH) -> PreprocessingPipeline:
        """
        Creates a preprocessing pipeline that removes stopwords, meaningless words,
        lowercase the text, removes diacritics, and applies
        stemming to the text. 

        Args:
            stemmer_path (str): Path to the stemming executable.

        Returns:
            PreprocessingPipeline: Preprocessing pipeline.
        """
        return PipelineBuilder() \
            .add_remove_stopwords() \
            .add_remove_meaningless() \
            .add_lowercase() \
            .add_remove_diacritics() \
            .add_stemming(stemmer_path) \
            .build()

    @staticmethod
    def create_lemmatization_pipeline(model_path: str = config.UDPIDE_MODEL_PATH) -> PreprocessingPipeline:
        """
        Creates a preprocessing pipeline that removes stopwords, meaningless words,
        lowercase the text, removes diacritics, and applies
        lemmatization to the text.

        Args:
            model_path (str): Path to the UDPipe model.

        Returns:
            PreprocessingPipeline: Preprocessing pipeline. 
        """
        return PipelineBuilder() \
            .add_remove_stopwords() \
            .add_remove_meaningless() \
            .add_lowercase() \
            .add_lemmatization(model_path) \
            .add_remove_diacritics() \
            .build()
    
    @staticmethod
    def create_simple_pipeline() -> PreprocessingPipeline:
        """
        Creates a simple preprocessing pipeline that removes stopwords, meaningless words,
        lowercase the text, and removes diacritics.

        Returns:
            PreprocessingPipeline: Preprocessing pipeline. 
        """
        return PipelineBuilder() \
            .add_remove_stopwords() \
            .add_remove_meaningless() \
            .add_lowercase() \
            .add_remove_diacritics() \
            .build()