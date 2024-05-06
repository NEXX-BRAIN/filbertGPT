from logi_langchain.chat_models import ChatOpenAI

from filbertgpt.utils.config_loader import load_config

openai_config = load_config()['llm']['openai']

gpt_4_turbo = ChatOpenAI(
    model_name=openai_config['model-name'],
    openai_api_key=openai_config['api-key'],
    temperature=0,
)

gpt_35_turbo = ChatOpenAI(
    model_name='gpt-3.5-turbo',
    openai_api_key=openai_config['api-key'],
    temperature=0,
)


def load_gpt_4_turbo():
    return gpt_4_turbo


def load_gpt_35_turbo():
    return gpt_35_turbo
