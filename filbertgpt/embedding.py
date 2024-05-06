# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2023/10/11 13:40
# @Author   : Perye (Pengyu) LI
# @FileName : embedding.py
# @Software : PyCharm

from logi_langchain.embeddings import OpenAIEmbeddings

from filbertgpt.utils.config_loader import load_config

embedding_config = load_config()['embedding']['openai']

embed = OpenAIEmbeddings(
    model=embedding_config['model-name'],
    openai_api_key=embedding_config['api-key']
)
