from logi_langchain.memory import CombinedMemory

from filbertgpt.memory.pinecone_memory import load_pinecone_memory
from filbertgpt.memory.conversation_memory import load_conversation_buffer_memory
from filbertgpt.memory.faiss_memory import load_faiss_memory

memory_suffix = """
Some knowledge may useful:
{history}

(You should carefully read them, but you do not need to use these pieces of information if not relevant)

Previous conversation history:
{conversation_history}
(You must read them thoroughly use them when the user is asking a question based on the conversation history, 
which is useful when you find the current question is not so clear or contains words like 'former', 'above')
"""

memory_input_variables = ['history', 'conversation_history']


def load_chain_memory():
    memories = [load_pinecone_memory(), load_conversation_buffer_memory()]
    return CombinedMemory(memories=memories)
