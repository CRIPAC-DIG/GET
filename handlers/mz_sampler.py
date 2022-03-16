"""
Module containing functions for negative item sampling.
"""

import numpy as np
from scipy.sparse import csr_matrix
import torch_utils
import time
import interactions


class Sampler(object):
    def __init__(self):
        super(Sampler, self).__init__()

    def get_train_instances(self, interactions, num_negatives: int):
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}
        Parameters
        ----------
        interactions: :class:`matchzoo.DataPack`
            training instances, used for generate candidates. Note that
            since I am using MatchZoo datapack, there are negative cases in left-right relation ship as
            well.
        num_negatives: int
            total number of negatives to sample for each sequence
        """

        query_ids = interactions.pos_queries.astype(np.int64)  # may not be unique
        query_contents = interactions.np_query_contents.astype(np.int64)
        query_lengths = interactions.np_query_lengths.astype(np.int64)

        doc_ids = interactions.pos_docs.astype(np.int64)
        doc_contents = interactions.np_doc_contents.astype(np.int64)
        doc_lengths = interactions.np_doc_lengths.astype(np.int64)

        negative_samples = np.zeros((query_ids.shape[0], num_negatives, interactions.padded_doc_length), np.int64)
        negative_samples_lens = np.zeros((query_ids.shape[0], num_negatives), np.int64)
        negative_docs_ids = np.zeros((query_ids.shape[0], num_negatives), np.int64)
        self._candidate = interactions.negatives

        for i, u in enumerate(query_ids):
            for j in range(num_negatives):
                x = self._candidate[u]
                neg_item = x[np.random.randint(len(x))]  # int
                # print("Neg_item: ", neg_item)
                neg_item_content = interactions.dict_doc_contents[neg_item]  # np.array
                negative_samples[i, j] = neg_item_content
                negative_samples_lens[i, j] = interactions.dict_doc_lengths[neg_item]
                negative_docs_ids[i, j] = neg_item
            # if u <= 0:
            #     print("Negative samples: ", negative_samples[i])
        # print(negative_samples)
        return query_ids, query_contents, query_lengths, \
               doc_ids, doc_contents, doc_lengths, \
               negative_docs_ids, negative_samples, negative_samples_lens

    def get_train_instances_declare(self, interactions: interactions.ClassificationInteractions,
                                    fixed_num_evidences: int):
        """
        ----------
        interactions: :class:`interactions.ClassificationInteractions`
            training instances,
        fixed_num_evidences: `int`
            fixed number of evidences for each claim
        """
        claim_sources = np.array([interactions.dict_claim_source[e] for e in interactions.claims])
        evidence_sources = np.array([interactions.dict_evd_source[e] for e in interactions.evidences])
        return interactions.claims, interactions.claims_contents, interactions.claims_lens, claim_sources, \
               interactions.evidences, interactions.evd_contents, interactions.evd_lens, evidence_sources, \
               interactions.pair_labels

    def get_train_instances_hanfc(self, interactions: interactions.ClassificationInteractions,
                                  fixed_num_evidences: int):
        """
        For each query/claim, we get its x number of evidences.
        Parameters
        ----------
        interactions: :class:`interactions.ClassificationInteractions`
            training instances,
        fixed_num_evidences: `int`
            fixed number of evidences for each claim
        """

        query_ids = interactions.claims.astype(np.int64)  # must be all unique
        query_labels = interactions.claims_labels
        query_contents = interactions.np_query_contents.astype(np.int64)
        query_lengths = interactions.np_query_lengths.astype(np.int64)
        query_sources = np.array([interactions.dict_claim_source[q] for q in query_ids])

        evd_docs_ids = np.zeros((query_ids.shape[0], fixed_num_evidences), np.int64) - 1  # all indices are -1
        # by default it is all pad tokens
        evd_docs_contents = np.zeros((query_ids.shape[0], fixed_num_evidences, interactions.padded_doc_length),
                                     np.int64)
        evd_docs_lens = np.zeros((query_ids.shape[0], fixed_num_evidences), np.int64)
        evd_sources = np.zeros((query_ids.shape[0], fixed_num_evidences), np.int64) - 1  # for padding sources are -1
        evd_cnt_each_query = np.zeros((query_ids.shape[0]), np.int64)

        for i, u in enumerate(query_ids):
            evidences_info = interactions.dict_claims_and_evidences_test[u]  # use u not i
            assert len(evidences_info) <= fixed_num_evidences
            evd_cnt_each_query[i] = len(evidences_info[0])  # number of real evidences for the query i
            # we have a list of evidences, now I need to take the content and doc_id
            for idx, (doc_id, doc_label, doc_content, doc_len) in enumerate(zip(*evidences_info)):
                evd_docs_contents[i][idx] = doc_content  # we already pad the content array with zeros due to init
                evd_docs_lens[i][idx] = doc_len  # we set 0 length for padding evidences
                evd_docs_ids[i][idx] = doc_id  # we set -1 as index for padding evidences
                evd_sources[i][idx] = interactions.dict_evd_source[doc_id][0]  # -1 since we have an array size 1

        return query_ids, query_contents, query_lengths, query_sources, \
               evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources, evd_cnt_each_query, query_labels

    def get_train_instances_char_man(self, interactions: interactions.ClassificationInteractions,
                                     fixed_num_evidences: int):
        """
        For each query/claim, we get its x number of evidences.
        Parameters
        ----------
        interactions: :class:`interactions.ClassificationInteractions`
            training instances,
        fixed_num_evidences: `int`
            fixed number of evidences for each claim
        """

        query_ids = interactions.claims.astype(np.int64)  # must be all unique
        query_labels = interactions.claims_labels
        query_contents = interactions.np_query_contents.astype(np.int64)
        query_lengths = interactions.np_query_lengths.astype(np.int64)
        query_char_source = interactions.np_query_char_source.astype(np.int64)
        query_sources = np.array([interactions.dict_claim_source[q] for q in query_ids])
        # query_adj = np.zeros((query_ids.shape[0], query_contents.shape[1], query_contents.shape[1]), np.int64)
        # query_adj = interactions.np_query_adj.astype(np.int64)
        query_adj = interactions.np_query_adj

        evd_docs_ids = np.zeros((query_ids.shape[0], fixed_num_evidences), np.int64) - 1  # all indices are -1
        # by default it is all pad tokens
        evd_docs_contents = np.zeros((query_ids.shape[0], fixed_num_evidences, interactions.padded_doc_length),
                                     np.int64)
        evd_docs_lens = np.zeros((query_ids.shape[0], fixed_num_evidences), np.int64)
        evd_sources = np.zeros((query_ids.shape[0], fixed_num_evidences), np.int64) - 1  # for padding sources are -1
        evd_cnt_each_query = np.zeros((query_ids.shape[0]), np.int64)
        evd_docs_char_source_contents = np.zeros((query_ids.shape[0], fixed_num_evidences,
                                                  interactions.padded_doc_char_source_length), np.int64)
        evd_docs_adj = np.zeros((query_ids.shape[0], fixed_num_evidences, interactions.padded_doc_length,
                                 interactions.padded_doc_length), np.float)

        for i, u in enumerate(query_ids):
            evidences_info = interactions.dict_claims_and_evidences_test[u]  # use u not i
            assert len(evidences_info) <= fixed_num_evidences
            evd_cnt_each_query[i] = len(evidences_info[0])  # number of real evidences for the query i
            # we have a list of evidences, now I need to take the content and doc_id
            for idx, (doc_id, doc_label, doc_content, doc_len, docs_adj) in enumerate(zip(*evidences_info)):
                evd_docs_contents[i][idx] = doc_content  # we already pad the content array with zeros due to init
                evd_docs_lens[i][idx] = doc_len  # we set 0 length for padding evidences
                evd_docs_ids[i][idx] = doc_id  # we set -1 as index for padding evidences
                evd_sources[i][idx] = interactions.dict_evd_source[doc_id][0]  # -1 since we have an array size 1
                evd_docs_char_source_contents[i][idx] = interactions.dict_char_right_src[doc_id]
                evd_docs_adj[i][idx] = interactions.dict_doc_adj[doc_id]
                # for row in range(doc_len):
                #     evd_docs_adj[i, idx, row, max(0, row-2):min(row+3, doc_len)] = 1
                #     for col in range(row+3, doc_len):
                #         evd_docs_adj[i][idx][row][col] = evd_docs_adj[i][idx][col][row] = \
                #             (evd_docs_contents[i, idx, row] == evd_docs_contents[i, idx, col])

        # for b in range(len(query_ids)):
        #     l = query_lengths[b]
        #     for i in range(l):
        #         query_adj[b, i, max(0, i - 2):min(i + 3, l)] = 1
        #         for j in range(i + 3, l):
        #             query_adj[b][i][j] = query_adj[b][j][i] = (query_contents[b][i] == query_contents[b][j])

        return query_ids, query_contents, query_lengths, query_sources, query_char_source, query_adj,\
               evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources, evd_cnt_each_query, \
               evd_docs_char_source_contents, query_labels, evd_docs_adj
