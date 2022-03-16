import typing
import csv

# import keras
import pandas as pd
import matchzoo
import os
from Models.FCWithEvidences.DeClare import pack
import torch
from typing import List, Dict
from tqdm import tqdm
import numpy as np

_url = "https://download.microsoft.com/download/E/5/F/" \
       "E5FCFCEE-7005-4814-853D-DAA7C66507E0/WikiQACorpus.zip"


def load_data(
    data_root: str,
    stage: str = 'train',
    task: str = 'classification',
    filtered: bool = False,
    return_classes: bool = False,
    kfolds: int = 5,
    extend_claim: bool = False
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load WikiQA data.
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filtered: Whether remove the questions without correct answers.
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.
    :param kfolds: `int` the number of folds
    :param extend_claim: `bool` `True` to merge claim id and claim text as a way to extend text of claims
    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ['dev'] + ["train_%s" % i for i in range(kfolds)] + ["test_%s" % i for i in range(kfolds)]:
        raise ValueError("%s is not a valid stage. Must be one of `train`, `dev`, and `test`." % stage)

    # data_root = _download_data()
    data_root = data_root
    file_path = os.path.join(data_root, '%s.tsv' % (stage))
    data_pack = _read_data(file_path, extend_claim)

    if task == 'ranking':
        task = matchzoo.tasks.Ranking()
    if task == 'classification':
        task = matchzoo.tasks.Classification()

    if isinstance(task, matchzoo.tasks.Ranking):
        return data_pack
    elif isinstance(task, matchzoo.tasks.Classification):
        # data_pack.one_hot_encode_label(task.num_classes, inplace=True)
        # I removed this due to PyTorch https://github.com/pytorch/pytorch/issues/5554
        if return_classes:
            return data_pack, [False, True]
        else:
            return data_pack
    else:
        raise ValueError("{%s} is not a valid task." % task + " " 
                         "Must be one of `Ranking` and `Classification`.")


def _read_data(path, extend_claim: bool = False):
    table = pd.read_csv(path, sep='\t', header=0, quoting=csv.QUOTE_NONE)

    def str2num(lab):
        # convert bool label to int
        lab = str(lab) if type(lab) == bool else lab
        assert lab == "TRUE" or lab == "True" or lab == "False" or lab == "FALSE" or lab == "MIXED"
        if lab == "TRUE" or lab == "True":
            return 1
        elif lab == "FALSE" or lab == "False":
            return 0
        else:
            return 2

    def merge_text(a, b):
        a = a.replace(".json", " ")
        a = " ".join(a.split("_"))
        a = " ".join(a.split("-"))
        return a + " " + b
    # cred_label	claim_id	claim_text	claim_source	evidence	evidence_source
    df = pd.DataFrame({
        'text_left': table['claim_text'],
        'raw_text_left': table['claim_text'].copy(),
        'claim_id': table["claim_id"],
        "claim_source": table["claim_source"],
        "char_claim_source": table["claim_source"].copy(),
        "raw_claim_source": table["claim_source"].copy(),
        "extended_text": table.progress_apply(lambda x: merge_text(x.claim_id, x.claim_text), axis = 1),

        'text_right': table['evidence'],
        'raw_text_right': table['evidence'].copy(),
        "evidence_source": table["evidence_source"],
        "char_evidence_source": table["evidence_source"].copy(),
        "raw_evidence_source": table["evidence_source"].copy(),

        'id_left': table['id_left'],
        'id_right': table['id_right'],
        'label': table['cred_label'].progress_apply(str2num)
    })
    if extend_claim:
        df["text_left"] = df["extended_text"]
        df["raw_text_left"] = df["extended_text"]
    # I decided to create a new datapack to avoid touching old code of learning to rank
    return pack.pack(df, selected_columns_left = ['text_left', 'id_left', 'raw_text_left', 'claim_source', 'raw_claim_source', 'char_claim_source'],
                     selected_columns_right = ['text_right', 'id_right', 'raw_text_right', 'evidence_source', 'raw_evidence_source', 'char_evidence_source'])

