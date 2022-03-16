

class KeyWordSettings(object):

    Doc_cID = "doc_cid"
    Doc_URL = "doc_ciurl"
    Doc_cLabel = "doc_clabel"
    Doc_wImages = "doc_wimages"
    Doc_wContent = "doc_wcontent"
    Relevant_Score = "relevant_score"

    Query_id = "qid"
    Query_TweetID = "qtweetid"
    Query_Images = "query_images"
    Ranked_Docs = "ranked_docs"
    Query_Content = "query_content"
    Query_Adj = "query_adj"
    Evd_Docs_Adj = "docs_adj"
    GNN_Window = "gnn_window"

    Query_lens = "query_lens"
    Doc_lens = "docs_lens"

    # for lstm keywords
    QueryLensIndices = "query_lens_indices"
    DocLensIndices = "doc_lens_indices"

    QueryIDs = "query_ids"
    DocIDs = "doc_ids"
    UseVisual = "use_visual"

    OutputRankingKey = "output_ranking"

    QueryCountVal = [1116, 1000, 187, 1159]
    QueryCountTest = [1001, 1164, 1118, 187, 156, 1160, 1500]

    UseCuda = "use_cuda"
    QuerySources = "query_sources"
    DocSources = "doc_sources"
    TempLabel = "fc_labels"
    DocContentNoPaddingEvidence = "doc_content_without_padding_evidences"  # to avoid empty sequences to lstm
    QueryContentNoPaddingEvidence = "query_content_without_padding_evidences"
    ClaimEmbeddings = "claim_embeddings"
    EvidenceEmbeddings = "evidences_embeddings"

    EvidenceCountPerQuery = "evd_cnt_each_query"
    FIXED_NUM_EVIDENCES = "fixed_num_evidences"

    LOSS_FUNCTIONS = ("cross_entropy")

    ClaimCountVal = [433, 356]
    ClaimCountTest = [782, 781, 391, 390, 644, 642, 323, 321]

    AUC_metric = "auc"
    F1_macro = "f1_macro"
    F1_micro = "f1_micro"
    F1 = "f1"
    PrecisionTrueCls = "precision_true_cls"
    RecallTrueCls = "recall_true_cls"
    F1TrueCls = "f1_true_cls"

    PrecisionFalseCls = "precision_false_cls"
    RecallFalseCls = "recall_false_cls"
    F1FalseCls = "f1_false_cls"

    PrecisionMixedCls = "precision_mixed_cls"
    RecallMixedCls = "recall_mixed_cls"
    F1MixedCls = "f1_mixed_cls"

    # for fact-checking error analysis
    class FCClass:
        DocAttentionScore = "doc_attention_score"
        WordAttentionScore = "word_attention_score"
        ClaimLabel = "claim_label"
        PredictedProb = "predicted_prob"
        AttentionWeightsInfo = "attention_weights_info"
        CharSourceKey = "char_source"
        QueryCharSource = "query_char_source"  # characters of claims' source (i.e. chars of speakers' names)
        DocCharSource = "doc_char_source"

    CLS_METRICS = [AUC_metric,F1_macro, F1_micro, F1,
                   PrecisionTrueCls, RecallTrueCls, F1TrueCls,
                   PrecisionFalseCls, RecallFalseCls, F1FalseCls,
                   PrecisionMixedCls, RecallMixedCls, F1MixedCls]

    OutputHandlerFactChecking = "output_handler_fact_checking"
