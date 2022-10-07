from abc import ABC
from enum import Enum


class ExplainerType(Enum):
    # TODO: More clear TransformerType is needed.
    raw = 1
    transform = 2
    approximate = 3
    transform_and_approximate = 4

class CaseImageExplainer(ABC):

    def __init__(self):
        super().__init__()