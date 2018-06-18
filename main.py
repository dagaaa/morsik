from file_parser import parse_wiki_dump, parse_dir_json
from search_engine import SearchEngine
from preprocess import Preprocessor, WithUrlPreprocessor
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from tfidf_model import *
tfidf= TFIDF(save=True)
# lda = LDA.with_url_handling(max_workers=4, save=True, num_topics=200)
# # jsons = parse_dir_json('/home/robert/PycharmProjects/mors_crawler-master/mors_crawler/data')
jsons = parse_dir_json('C:\\Users\\dagmara\\Desktop\\mobileworld\\data', limit=100)
urls, documents = zip(*jsons)
# preprocessor = Preprocessor()
# preprocessed_docs = preprocessor.process_docs(documents)

# dct = Dictionary(preprocessed_docs)
# corpus = [dct.doc2bow(line) for line in preprocessed_docs]  # convert dataset to BoW format
# model = TfidfModel(corpus)  # fit model
# vector = model[corpus[0]]  # apply model
model, dict = tfidf.train(documents)
searchEngine = SearchEngine()
#
# yy = searchEngine.infer_all(jsons)
print(len(searchEngine.dictionary))
searchEngine.dummy_index(jsons)

aa = input('type query:\n')

while aa != 'q':
    print(searchEngine.dummy_search(aa)[:20])
    aa = input('type query:\n')
