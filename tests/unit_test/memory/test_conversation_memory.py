# ./venv/bin/python
# _*_ coding: utf-8 _*_
# @Time     : 2024/1/29 13:39
# @Author   : Perye (Pengyu) LI
# @FileName : test_conversation_memory.py
import uuid
from unittest import TestCase

from logi_langchain.conf import context_auto_saving

from filbertgpt.chain import load_sql_chain
from filbertgpt.memory.conversation_memory import load_conversation_mongo_memory


# @Software : PyCharm
class Test(TestCase):
    def test_load_conversation_mongo_memory(self):
        chat_memory = load_conversation_mongo_memory(str(uuid.uuid4()))
        assert len(chat_memory.chat_memory.messages) == 0
        chain = load_sql_chain(memory=chat_memory)
        chain({'input': 'how many goods do we have?'})
        chain({'input': 'how many goods do we have'})
        assert len(chat_memory.chat_memory.messages) == 4
