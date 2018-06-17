from typing import List

from nltk import PorterStemmer, RegexpTokenizer
from stop_words import get_stop_words

# from myparser import timed
from tfidf_types import UrlDoc, PreprocessedUrlDoc, Doc, PreprocessedDoc


class Preprocessor(object):
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.en_stopwords = set(get_stop_words('en'))
        self.p_stemmer = PorterStemmer()

    def preprocess_doc(self, doc: Doc) -> PreprocessedDoc:
        tokens = self.tokenizer.tokenize(doc.lower())

        stopped_tokens = [i for i in tokens if i not in self.en_stopwords]

        stemmed_tokens = [self.p_stemmer.stem(i) for i in stopped_tokens]

        return stemmed_tokens

    # @timed
    def process_docs(self, doc_list: List[Doc]) -> List[PreprocessedDoc]:
        return [self.preprocess_doc(doc) for doc in doc_list]

    def preprocess_doc_with_url(self, doc_with_url: UrlDoc) -> PreprocessedUrlDoc:
        url, content = doc_with_url

        return url, self.preprocess_doc(content)

    # @timed
    def process_docs_with_urls(self, urldoc_list: List[UrlDoc]) -> List[PreprocessedUrlDoc]:
        return [self.preprocess_doc_with_url(urldoc) for urldoc in urldoc_list]



class WithUrlPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def preprocess_doc(self, doc: UrlDoc) -> PreprocessedDoc:
        _, content = doc
        return super().preprocess_doc(content)
