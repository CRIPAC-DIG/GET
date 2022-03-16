"""Convert list of input into class:`DataPack` expected format."""

import typing

import pandas as pd
import numpy as np
from matchzoo.data_pack.pack import _gen_ids
import matchzoo
from typing import List


def pack(df: pd.DataFrame,
         selected_columns_left: List[str],
         selected_columns_right: List[str]) -> 'matchzoo.DataPack':
    """
    This is a customized class to avoid overriding code of Learning to Rank project
    Pack a :class:`DataPack` using `df`.

    The `df` must have `text_left` and `text_right` columns. Optionally,
    the `df` can have `id_left`, `id_right` to index `text_left` and
    `text_right` respectively. `id_left`, `id_right` will be automatically
    generated if not specified.

    :param df: Input :class:`pandas.DataFrame` to use.
    :param selected_columns_left: `List[str]` a list of columns selected for building data pack
    :param selected_columns_right: `List[str]` a list of columns selected for building data pack

    Examples::
        >>> import matchzoo as mz
        >>> import pandas as pd
        >>> df = pd.DataFrame(data={'text_left': list('AABC'),
        ...                         'text_right': list('abbc'),
        ...                         'label': [0, 1, 1, 0]})
        >>> mz.pack(df).frame()
          id_left text_left id_right text_right  label
        0     L-0         A      R-0          a      0
        1     L-0         A      R-1          b      1
        2     L-1         B      R-1          b      1
        3     L-2         C      R-2          c      0

    """
    if 'text_left' not in df or 'text_right' not in df:
        raise ValueError(
            'Input data frame must have `text_left` and `text_right`.')

    # Gather IDs
    if 'id_left' not in df:
        id_left = _gen_ids(df, 'text_left', 'L-')
    else:
        id_left = df['id_left']
    if 'id_right' not in df:
        id_right = _gen_ids(df, 'text_right', 'R-')
    else:
        id_right = df['id_right']

    # Build Relation
    relation = pd.DataFrame(data={'id_left': id_left, 'id_right': id_right})
    for col in df:
        if col not in ['id_left', 'id_right', 'text_left', 'text_right']:
            relation[col] = df[col]

    # Build Left and Right
    left = _merge(df, id_left, selected_columns_left)
    right = _merge(df, id_right, selected_columns_right)
    return matchzoo.DataPack(relation, left, right)


def _merge(data: pd.DataFrame, ids: typing.Union[list, np.array], fields: List[str]):
    text_label, id_label = fields[0], fields[1]
    info = [(text_label, data[text_label]), (id_label, ids)] + [(e, data[e]) for e in fields[2:]]
    info = dict(info)
    left = pd.DataFrame(data=info)
    left.drop_duplicates(id_label, inplace=True)
    left.set_index(id_label, inplace=True)
    return left
