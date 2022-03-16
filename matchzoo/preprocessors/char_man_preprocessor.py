"""Basic Preprocessor."""

from tqdm import tqdm

from . import units
from matchzoo import DataPack
from matchzoo.engine.base_preprocessor import BasePreprocessor
from .build_vocab_unit import build_vocab_unit
from .build_unit_from_data_pack import build_unit_from_data_pack
from .chain_transform import chain_transform
from .units import Vocabulary
from .units import StatefulUnit
tqdm.pandas()


class CharManPreprocessor(BasePreprocessor):
    """
    Preprocessor for model Character Multiperspective Attention Network (CharMAN)
    Char-MAN preprocessor helper which has source embeddings. Add char embeddings for characters

    :param fixed_length_left: Integer, maximize length of :attr:`left` in the
        data_pack.
    :param fixed_length_right: Integer, maximize length of :attr:`right` in the
        data_pack.
    :param filter_mode: String, mode used by :class:`FrequenceFilterUnit`, Can
        be 'df', 'cf', and 'idf'.
    :param filter_low_freq: Float, lower bound value used by
        :class:`FrequenceFilterUnit`.
    :param filter_high_freq: Float, upper bound value used by
        :class:`FrequenceFilterUnit`.
    :param remove_stop_words: Bool, use :class:`StopRemovalUnit` unit or not.

    Example:
        >>> import matchzoo as mz
        >>> train_data = mz.datasets.toy.load_data('train')
        >>> test_data = mz.datasets.toy.load_data('test')
        >>> preprocessor = mz.preprocessors.BasicPreprocessor(
        ...     fixed_length_left=10,
        ...     fixed_length_right=20,
        ...     filter_mode='df',
        ...     filter_low_freq=2,
        ...     filter_high_freq=1000,
        ...     remove_stop_words=True
        ... )
        >>> preprocessor = preprocessor.fit(train_data, verbose=0)
        >>> preprocessor.context['input_shapes']
        [(10,), (20,)]
        >>> preprocessor.context['vocab_size']
        225
        >>> processed_train_data = preprocessor.transform(train_data,
        ...                                               verbose=0)
        >>> type(processed_train_data)
        <class 'matchzoo.data_pack.data_pack.DataPack'>
        >>> test_data_transformed = preprocessor.transform(test_data,
        ...                                                verbose=0)
        >>> type(test_data_transformed)
        <class 'matchzoo.data_pack.data_pack.DataPack'>

    """

    def __init__(self, fixed_length_left: int = 30,
                 fixed_length_right: int = 30,
                 fixed_length_left_src: int = 30,
                 fixed_length_right_src: int = 30,
                 filter_mode: str = 'df',
                 filter_low_freq: float = 2,
                 filter_high_freq: float = float('inf'),
                 remove_stop_words: bool = False):
        """Initialization."""
        super().__init__()
        self._fixed_length_left = fixed_length_left
        self._fixed_length_right = fixed_length_right
        self._fixed_length_left_src = fixed_length_left_src
        self._fixed_length_right_src = fixed_length_right_src
        self._left_fixedlength_unit = units.FixedLength(
            self._fixed_length_left,
            pad_mode='post'
        )
        self._right_fixedlength_unit = units.FixedLength(
            self._fixed_length_right,
            pad_mode='post'
        )
        # for padding character level of left_source and right_source
        self._left_char_src_fixedlength_unit = units.FixedLength(self._fixed_length_left_src, pad_mode='post')
        self._right_char_src_fixedlength_unit = units.FixedLength(self._fixed_length_right_src, pad_mode='post')

        self.char_unit = units.ngram_letter.NgramLetter(ngram=1, reduce_dim=True)
        self._units = self._default_units()
        if remove_stop_words:
            self._units.append(units.stop_removal.StopRemoval())

    def fit(self, data_pack: DataPack, verbose: int = 1):
        """
        Fit pre-processing context for transformation.

        :param data_pack: data_pack to be preprocessed.
        :param verbose: Verbosity.
        :return: class:`BasicPreprocessor` instance.
        """
        data_pack = data_pack.apply_on_text(chain_transform(self._units),
                                            verbose=verbose)
        # fitted_filter_unit = build_unit_from_data_pack(self._filter_unit,
        #                                                data_pack,
        #                                                flatten=False,
        #                                                mode='right',
        #                                                verbose=verbose)
        # data_pack = data_pack.apply_on_text(fitted_filter_unit.transform,
        #                                     mode='right', verbose=verbose)
        # self._context['filter_unit'] = fitted_filter_unit

        vocab_unit = build_vocab_unit(data_pack, verbose=verbose)
        self._context['vocab_unit'] = vocab_unit

        vocab_size = len(vocab_unit.state['term_index'])  # + 1  # +1 for padding
        self._context['vocab_size'] = vocab_size
        self._context['embedding_input_dim'] = vocab_size
        self._context['input_shapes'] = [(self._fixed_length_left,),
                                         (self._fixed_length_right,)]

        claim_source_unit = build_entity_unit(column = "claim_source", data_pack = data_pack, mode = "left")
        article_source_unit = build_entity_unit(column = "evidence_source", data_pack = data_pack, mode = "right")
        self._context['claim_source_unit'] = claim_source_unit
        self._context['article_source_unit'] = article_source_unit

        char_source_unit = build_ngram_unit(left_column = "claim_source", right_column="evidence_source",
                                            data_pack = data_pack, mode = "both")
        self._context['char_source_unit'] = char_source_unit

        return self

    def transform(self, data_pack: DataPack, verbose: int = 1) -> DataPack:
        """
        Apply transformation on data, create fixed length representation.

        :param data_pack: Inputs to be preprocessed.
        :param verbose: Verbosity.

        :return: Transformed data as :class:`DataPack` object.
        """
        data_pack = data_pack.copy()

        def map_claim_source(entity: str): return self._context['claim_source_unit'].transform([entity])

        def map_evidence_source(entity: str): return self._context['article_source_unit'].transform([entity])

        def map_src2char(entity: str):
            return self._context['char_source_unit'].transform(list(entity))

        data_pack.left["claim_source"] = data_pack.left["claim_source"].progress_apply(map_claim_source)
        data_pack.left["char_claim_source"] = data_pack.left["char_claim_source"].progress_apply(map_src2char)
        data_pack.left["char_claim_source"] = data_pack.left["char_claim_source"].progress_apply(
            self._left_char_src_fixedlength_unit.transform)

        data_pack.right["evidence_source"] = data_pack.right["evidence_source"].progress_apply(map_evidence_source)
        data_pack.right["char_evidence_source"] = data_pack.right["char_evidence_source"].progress_apply(map_src2char)
        data_pack.right["char_evidence_source"] = data_pack.right["char_evidence_source"].progress_apply(
            self._right_char_src_fixedlength_unit.transform)

        data_pack.apply_on_text(chain_transform(self._units), inplace=True, verbose=verbose)

        # data_pack.apply_on_text(self._context['filter_unit'].transform,
        #                         mode='right', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._context['vocab_unit'].transform,
                                mode='both', inplace=True, verbose=verbose)
        data_pack.append_text_length(inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._left_fixedlength_unit.transform,
                                mode='left', inplace=True, verbose=verbose)
        data_pack.apply_on_text(self._right_fixedlength_unit.transform,
                                mode='right', inplace=True, verbose=verbose)

        max_len_left = self._fixed_length_left
        max_len_right = self._fixed_length_right

        data_pack.left['length_left'] = \
            data_pack.left['length_left'].apply(
                lambda val: min(val, max_len_left))

        data_pack.right['length_right'] = \
            data_pack.right['length_right'].apply(
                lambda val: min(val, max_len_right))

        return data_pack


def build_entity_unit(
    column: str,
    data_pack: DataPack,
    mode: str = 'both',
    verbose: int = 1
) -> Vocabulary:
    """
    Build a :class:`preprocessor.units.Vocabulary` given `data_pack`.

    The `data_pack` should be preprocessed forehand, and each item in
    `text_left` and `text_right` columns of the `data_pack` should be a list
    of tokens.

    :param column: `str` the selected column to build units
    :param data_pack: The :class:`DataPack` to build vocabulary upon.
    :param mode: One of 'left', 'right', and 'both', to determine the source
    data for building the :class:`VocabularyUnit`.
    :param verbose: Verbosity.
    :return: A built vocabulary unit.

    """
    unit = Vocabulary()
    corpus = []
    def func(entity: str): corpus.append(entity.strip())
    assert mode in ["left", "right"]
    if mode == "left":
        data_pack.left[column].progress_apply(func)
    elif mode == "right":
        data_pack.right[column].progress_apply(func)
    else:
        raise NotImplemented("Not coded for both columns")

    if verbose:
        description = 'Building Entities ' + unit.__class__.__name__ + ' from a datapack.'
        corpus = tqdm(corpus, desc=description)
    unit.fit(corpus)
    return unit


def build_ngram_unit(left_column: str, right_column: str, data_pack: DataPack, mode: str = 'both', verbose: int = 1):
    """
    Build a :class:`preprocessor.units.Vocabulary` given `data_pack`.

    The `data_pack` should be preprocessed forehand, and each item in
    `text_left` and `text_right` columns of the `data_pack` should be a list
    of tokens.

    :param column: `str` the selected column to build units
    :param data_pack: The :class:`DataPack` to build vocabulary upon.
    :param mode: One of 'left', 'right', and 'both', to determine the source
    data for building the :class:`VocabularyUnit`.
    :param verbose: Verbosity.
    :return: A built vocabulary unit.

    """
    unit = Vocabulary()
    corpus = []

    def func(entity: str):
        assert type(entity) == str
        entity = entity.strip()
        for c in entity: corpus.append(c)

    assert mode == "both"
    data_pack.left[left_column].progress_apply(func)
    data_pack.right[right_column].progress_apply(func)

    if verbose:
        description = 'Building Characters ' + unit.__class__.__name__ + ' from a datapack.'
        corpus = tqdm(corpus, desc=description)
    unit.fit(corpus)
    return unit
