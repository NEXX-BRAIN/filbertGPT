# ./venv/Scripts/python
# _*_ coding: utf-8 _*_
# @Time     : 2023/7/21 10:40 AM
# @Author   : Perye (Pengyu) LI
# @FileName : config_loader.py
# @Software : PyCharm

import re
import traceback
from pathlib import Path
import yaml


def load_config():
    caller_path = str(traceback.extract_stack()[-2]).split(',')[0].lstrip('<FrameSummary file ')
    print(caller_path)
    module_path = (
        re.match('(.+)/filbertgpt/', caller_path) or
        re.match('(.+)\\\\filbertgpt\\\\', caller_path)
    ).group()
    with open(Path(module_path) / 'config.yml', 'r') as config:
        return yaml.safe_load(config)
