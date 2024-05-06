# ./venv/bin/python
# _*_ coding: utf-8 _*_
# @Time     : 2024/1/8 10:03
# @Author   : Perye (Pengyu) LI
# @FileName : pydantic_utils.py
# @Software : PyCharm
import abc
import os
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure


class DataCoder(abc.ABC):

    @abc.abstractmethod
    def encode_figure(self, fig: Figure) -> str:
        pass

    @abc.abstractmethod
    def encode_df(self, df: pd.DataFrame) -> str:
        pass

    @abc.abstractmethod
    def decode_figure(self, code: str) -> np.ndarray | BytesIO | str:
        """Monochrome image of shape (w,h) or (w,h,1) OR a color image of shape (w,h,3) OR an RGBA image of shape (w,
        h,4) OR a URL to fetch the image from OR a path of a local image file OR an SVG XML string like <svg
        xmlns=...</svg> OR a list of one of the above, to display multiple images."""
        pass

    @abc.abstractmethod
    def decode_df(self, code: str) -> pd.DataFrame:
        pass


class LocalFileCoder(DataCoder):

    cache_dir = f'{(os.environ.get("TMPDIR", "") or os.environ.get("TEMP", "") or os.environ.get("TMP", "") or ".cache/")}'

    def encode_figure(self, fig: Figure) -> str:
        path = Path(self.cache_dir) / f'{uuid.uuid4().hex}.png'
        fig.savefig(path)
        return str(path)

    def encode_df(self, df: pd.DataFrame) -> str:
        path = Path(self.cache_dir) / f'{uuid.uuid4().hex}.csv'
        df.to_csv(path)
        return str(path)

    def decode_figure(self, path: str) -> str:
        return path

    def decode_df(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)
