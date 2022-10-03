import os
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple, Union

import loguru
import numpy as np
import pandas as pd

class ExplainerType(Enum):
    # TODO: More clear TransformerType is needed.
    raw = 1
    transform = 2
    approximate = 3
    transform_and_approximate = 4

class CaseImageExplainer(ABC):

    def __init__(self):
        super().__init__()