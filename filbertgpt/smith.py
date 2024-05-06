# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2023/12/7 13:45
# @Author   : Perye (Pengyu) LI
# @FileName : langsmith.py
# @Software : PyCharm

from langsmith import Client
from filbertgpt.utils.config_loader import load_config

langsmith_config = load_config()['langsmith']

langsmith_project = langsmith_config['project']

langsmith_client = Client(api_key=langsmith_config['api-key'])
