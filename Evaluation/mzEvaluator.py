import numpy as np
from setting_keywords import KeyWordSettings
from typing import List, Dict


def compute_average_classification_results(kfold_results: List[Dict[str, float]], metrics: List[str], **kargs) \
        -> Dict[str, Dict[str, float]]:
    """
    Compute average results of every metric in list of metrics
    Parameters
    ----------
    kfold_results: a list of k-folds results where each element is a dict of metric
    metrics: a list of metrics string like `auc`, `f1-macros` and so on
    Returns
    ----------
        a dict of metrics where key is a metric, value is another dict with following structure {"avg": vl, "std": vl}
    """
    k = len(kfold_results)
    avg_results = {}
    for metric in metrics:
        ls = [fold[metric] for fold in kfold_results]
        avg, std = np.mean(ls), np.std(ls)
        avg_results[metric] = {"avg": avg, "std": std,
                               "all_%s_folds" % k: " ".join([str(e) for e in ls])}
    template = "\\begin{tabular}[c]{@{}l@{}} %.5f \\\\$\\pm$ %.5f \\end{tabular}"
    targets = [KeyWordSettings.AUC_metric ,KeyWordSettings.F1_macro, KeyWordSettings.F1_micro,
               KeyWordSettings.F1TrueCls, KeyWordSettings.PrecisionTrueCls, KeyWordSettings.RecallTrueCls,
               KeyWordSettings.F1FalseCls, KeyWordSettings.PrecisionFalseCls, KeyWordSettings.RecallFalseCls,
               KeyWordSettings.F1MixedCls, KeyWordSettings.PrecisionMixedCls, KeyWordSettings.RecallMixedCls]
    ls = ["ModelName"] + [template % (avg_results[key]["avg"], avg_results[key]["std"]) for key in targets]
    latex = "&".join(ls)
    output_handler = kargs[KeyWordSettings.OutputHandlerFactChecking]
    output_handler.myprint(latex)
    return avg_results
