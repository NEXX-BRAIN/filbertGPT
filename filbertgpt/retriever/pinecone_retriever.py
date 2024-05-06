# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2023/10/11 13:41
# @Author   : Perye (Pengyu) LI
# @FileName : pinecone_retriever.py
# @Software : PyCharm

from pinecone import Pinecone, PodSpec

from filbertgpt.embedding import embed
from logi_langchain.vectorstores import Pinecone as PineconeVectorstore
from logi_langchain.utils.logger import logi_logger

from filbertgpt.utils.config_loader import load_config


pinecone_config = load_config()['retriever']['pinecone']

# find API key in console at app.pinecone.io
YOUR_API_KEY = pinecone_config['api-key']
# find ENV (cloud region) next to API key in console
YOUR_ENV = pinecone_config['env']
# name of your index
INDEX_NAME = pinecone_config['index-name']

logi_logger.info(f'Initializing pinecone environment.')
pc = Pinecone(api_key=YOUR_API_KEY)

logi_logger.info(f'Detecting whether the index {INDEX_NAME} exists.')


def index_names():
    for idx in pc.list_indexes():
        yield idx['name']


if INDEX_NAME not in index_names():
    logi_logger.warning(f'The index {INDEX_NAME} does not exist. Creating now...')
    # we create a new index
    pc.create_index(
        name=INDEX_NAME,
        metric='dotproduct',
        dimension=1536,  # 1536 dim of text-embedding-ada-002
        spec=PodSpec(
            environment=YOUR_ENV
        )
    )

text_field = "text"

# switch back to normal index for langchain
index = pc.Index(INDEX_NAME)

vectorstore = PineconeVectorstore(
    index, embed, text_field
)

_pinecone_retriever = vectorstore.as_retriever(search_kwargs=pinecone_config['search_kwargs'])


def load_retriever():
    return _pinecone_retriever
