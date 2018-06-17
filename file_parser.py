import json
from typing import List

import os

# from myparser import timed
from tfidf_types import UrlDoc, Doc


def parse_wiki_dump(filename: str, limit: int = None) -> List[Doc]:
    texts = []
    buffer = []
    with open(filename) as f:
        for line in f.readlines():
            if line.startswith(" =") and line.endswith("= \n") and line.count('=') == 2:
                if buffer:
                    texts.append(''.join(buffer))
                    buffer = []
                    if limit and limit < len(texts):
                        return texts[1:]
            trimmed_line = line.replace('<unk>', '')
            buffer.append(trimmed_line)
        if buffer:
            texts.append(''.join(buffer))
    return texts[1:]


# @timed
def parse_dir_json(dirpath: str, limit=None) -> List[UrlDoc]:
    documents = []
    for file in os.listdir(dirpath):
        with open(dirpath + '/' + file, encoding="utf8", errors='ignore') as f:
            data = json.load(f)
            content = data['title'] + ' ' + ' '.join(data['description']) + ' ' + data['description'] + ' ' + data[
                'content']
            documents.append((data['url'], content))
        if limit is not None and len(documents) > limit:
            return documents
    return documents
