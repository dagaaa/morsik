from typing import List, Union

from concurrent.futures import ProcessPoolExecutor

from gensim.corpora import Dictionary
from gensim.models import TfidfModel

# from myparser import timed
from preprocess import Preprocessor, WithUrlPreprocessor
from tfidf_types import Doc, UrlDoc


class TFIDF(object):
    def __init__(self,
                 save: bool = False,
                 tfidf_filename: str = "tfidf_model",
                 dict_filename: str = "dictionary",
                 preprocessor=None):
        self.tfidf_filename = tfidf_filename
        self.dict_filename = dict_filename
        self.save = save
        self.preprocessor = preprocessor if preprocessor is not None else Preprocessor()

    # @timed
    def train(self, doc_list: List[Doc]) -> (TfidfModel, Dictionary):
        preprocessed_docs = self.preprocessor.process_docs(doc_list)

        dictionary = Dictionary(preprocessed_docs)
        corpus = [dictionary.doc2bow(line) for line in preprocessed_docs]  # convert dataset to BoW format
        model = TfidfModel(corpus)  # fit model

        if self.save:
            model.save(self.tfidf_filename)
            dictionary.save(self.dict_filename)

        return model, dictionary
