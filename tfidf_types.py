from typing import Tuple, List

Doc = str
Url = str
Token = str
UrlDoc = Tuple[Url, Doc]
PreprocessedDoc = List[Token]
PreprocessedUrlDoc = Tuple[Url, List[Token]]
