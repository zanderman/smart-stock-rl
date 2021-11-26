"""Generic RL agent class definition."""
from __future__ import annotations
import json
import numpy as np
import os
import zipfile
try:
    import zlib
    mode = zipfile.ZIP_DEFLATED
except:
    mode = zipfile.ZIP_STORED


class Agent:
    def __init__(self) -> None:
        super().__init__()


    @staticmethod
    def _save_to_file(path: str, data: dict = None, parameters: dict = None):
        """
        Save format is based off of StableBaselines implementation: https://stable-baselines.readthedocs.io/en/master/guide/save_format.html

        https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/base_class.py
        """

        # If path was a string, check extension and append ".zip" if omitted.
        if isinstance(path, str):
            _, ext = os.path.splitext(path)
            if ext == "":
                path = f"{path}.zip"

        # Create zip archive for content.
        with zipfile.ZipFile(path, 'w', mode) as zip:

            # Dump class parameters.
            if data is not None:
                with zip.open('data') as f:
                    json.dump(data, f)

            # Dump parameters as numpy compressed archive.
            if parameters is not None:
                with zip.open('parameters') as f:
                    np.savez(f, **parameters)


    @staticmethod
    def _load_from_file(path: str) -> tuple[dict, dict]:

        # Define initial outgoing values.
        data = None
        parameters = None

        # Create zip archive for content.
        with zipfile.ZipFile(path, 'r') as zip:

            # Get list of files in archive.
            namelist = zip.namelist()

            # Class member variables.
            if 'data' in namelist:
                with zip.open('data') as f:
                    data = json.load(f)

            # Parameter archive.
            if 'parameters' in namelist:
                with zip.open('parameters') as f:
                    parameters = np.load(f)

        return data, parameters


    def save(self, path: str):
        raise NotImplementedError


    def load(self, path: str):
        raise NotImplementedError
