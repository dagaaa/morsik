import sys

from configuration.tfidf.config import TfidfConfig
from search_engine.tfidf.logger.logger_config import init_logger

if __name__ == '__main__':
    init_logger()

    config = TfidfConfig(sys.argv[1],'tfidf_search_engine').get_current_config()

    docs = parse_dir_json(config['data_path'])

    searchEngine = SearchEngine(config['topics']) #todo to chyba nie powinno zwierac topics
    searchEngine.load_model(config['model_path'], config['dict_path'])
    searchEngine.dummy_index(docs)

    searchEngine.save_index(config['index_path'], config['url_path'])

    print(searchEngine.dummy_search("israel bank money")[:3])
    print(searchEngine.dummy_search("biggest wars in europe history")[:3])

    print(searchEngine.dummy_search("bitcoin and blockchain are future")[:3])

    print(searchEngine.dummy_search("gay marriages in europe")[:3])

    print(searchEngine.dummy_search("USA trump foreign policy")[:3])

