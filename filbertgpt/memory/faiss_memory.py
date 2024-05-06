# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2023/9/11 15:21
# @Author   : Perye (Pengyu) LI
# @FileName : faiss_memory.py
# @Software : PyCharm

"""
FAISS memory is used for caching chat history and retrieving related items,
should be created for every distinct user.
"""
from logi_langchain.memory.faiss_memory import FAISSMemory
from filbertgpt.retriever.faiss_retriever import load_retriever


def load_faiss_memory():
    faiss_memory = FAISSMemory(retriever=load_retriever(), memory_key='history', input_key='input')
    faiss_memory.clear()
    return faiss_memory
