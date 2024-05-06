# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2023/7/14 4:15 PM
# @Author   : Perye (Pengyu) LI
# @FileName : pinecone_memory.py
# @Software : PyCharm

"""
Pinecone memory is used for retrieving background knowledge, only need to be initialized once on project starting.
"""
import json
from pathlib import Path

from logi_langchain.memory import PineconeMemory
from logi_langchain.utils.logger import logi_logger, ch

from filbertgpt.retriever.pinecone_retriever import load_retriever, index

pinecone_memory = PineconeMemory(retriever=load_retriever(), memory_key='history', input_key='input')

logi_logger.info(f'Detecting whether the index has initialized with predefined vectors.')

if index.describe_index_stats()['total_vector_count']:
    logi_logger.info(f'Loading predefined vectors into pinecone.')
    with open(Path(__file__).parent / 'vectors.json', 'r', encoding='utf-8') as f:
        vectors = json.load(f)
        for v in vectors:
            pinecone_memory.init_context(*v)

logi_logger.info('Pinecone memory initialization finished.')

logi_logger.removeHandler(ch)


def load_pinecone_memory():
    return pinecone_memory
