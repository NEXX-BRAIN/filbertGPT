# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2023/10/11 13:42
# @Author   : Perye (Pengyu) LI
# @FileName : faiss_retriever.py
# @Software : PyCharm

from logi_langchain.vectorstores import FAISS

from filbertgpt.embedding import embed


def load_retriever():
    vectorstore = FAISS.from_texts([''], embed)
    return vectorstore.as_retriever(search_kwargs=dict(k=3))
