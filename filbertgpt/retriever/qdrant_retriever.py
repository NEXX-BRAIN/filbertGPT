# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2023/10/11 13:42
# @Author   : Perye (Pengyu) LI
# @FileName : qdrant_retriever.py
# @Software : PyCharm
from logi_langchain.vectorstores import Qdrant

from filbertgpt.utils.config_loader import load_config
from filbertgpt.embedding import embed
from filbertgpt.resources.wiki import wiki

embedding_config = load_config()['embedding']['openai']
qdrant_config = load_config()['retriever']['qdrant']


qdrant = Qdrant.from_texts(
    [f'{item[0]["input"]}: {item[1]["output"]}.' for item in wiki],
    embed,
    url=qdrant_config['url'],
    api_key=qdrant_config['api-key'],
    collection_name="logi_wiki",
    force_recreate=False,
)

_qdrant_retriever = qdrant.as_retriever(search_kwargs=qdrant_config['search_kwargs'])


def load_retriever():
    return _qdrant_retriever
