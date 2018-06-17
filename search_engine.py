import pickle
from typing import List

from gensim import similarities
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

# from myparser import timed
from preprocess import Preprocessor

# config file here?
from tfidf_types import UrlDoc


class SearchEngine(object):
    def __init__(self,
                 tfidf_model=None,
                 dictionary=None,
                 tfidf_path='tfidf_model',
                 dictionary_path='dictionary', ):
        self.tfidf_model = tfidf_model if tfidf_model is not None else TfidfModel.load(tfidf_path)
        self.dictionary = dictionary if dictionary is not None else Dictionary.load(dictionary_path)
        self.preprocessor = Preprocessor()
        self.index = None
        self.urls = None

    def infer(self, document: str) -> list:
        text = self.preprocessor.preprocess_doc(document)
        bow = self.dictionary.doc2bow(text)
        return self.tfidf_model[bow]

    def infer_all(self, docs_with_urls: List[UrlDoc]) -> list:
        preproc_docs_with_urls = self.preprocessor.process_docs_with_urls(docs_with_urls)
        bags_of_words = [(url, self.dictionary.doc2bow(doc)) for url, doc in preproc_docs_with_urls]
        return [(url, self.tfidf_model[bow]) for url, bow in bags_of_words]

    def dummy_index(self, docs_with_urls: List[UrlDoc]):
        urls, doc_bows = zip(*self.infer_all(docs_with_urls))
        self.urls = urls
        self.index = similarities.SparseMatrixSimilarity(doc_bows, num_features=len(self.dictionary))
        self.save('urls',self.urls)
        self.index.save('index')

    def save(self, filename: str, what):
        with open(filename, 'wb') as f:
            pickle.dump(what, f)

    def load(self, filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def load_index(self, path='index'):
        self.index = similarities.SparseMatrixSimilarity.load(path)
        self.load('urls')

    # @timed
    def dummy_search(self, query):
        infer_query=self.infer(query)
        inferred = self.index[infer_query]
        ss = sorted(enumerate(inferred), key=lambda item: -item[1])
        return [(self.urls[i], sim) for i, sim in ss]
