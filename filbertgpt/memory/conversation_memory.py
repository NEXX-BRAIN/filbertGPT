# ./venv/Scripts/python
# _*_ coding: utf-8 _*_

"""
Conversation memory is used for saving chat history,
should be created for every distinct user.
"""
from logi_langchain.memory import ConversationBufferMemory

from filbertgpt.utils.config_loader import load_config


def load_conversation_buffer_memory():
    return ConversationBufferMemory(
        human_prefix='User Question',
        ai_prefix='Your Answer',
        memory_key='conversation_history',
        input_key='input',
        output_key='output'
    )

