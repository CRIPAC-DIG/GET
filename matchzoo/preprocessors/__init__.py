from . import units
from .basic_preprocessor import BasicPreprocessor
from .bow_preprocessor import BoWPreprocessor
from .char_man_preprocessor import CharManPreprocessor


def list_available() -> list:
    from matchzoo.engine.base_preprocessor import BasePreprocessor
    from matchzoo.utils import list_recursive_concrete_subclasses
    return list_recursive_concrete_subclasses(BasePreprocessor)
