import torch
import numpy as np
import torch_utils
from Models import base_model
import torch_utils as my_utils
import time
import json
import interactions
from handlers.tensorboard_writer import TensorboardWrapper
from setting_keywords import KeyWordSettings
from Fitting.FittingFC.multi_level_attention_composite_fitter import MultiLevelAttentionCompositeFitter
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score
import sklearn


class CharManFitterQueryRepr1(MultiLevelAttentionCompositeFitter):
    """
    I implement this class for testing if the padding all zeros sequences are the root cause of performance reduction.
    """

    def __init__(self, net: base_model.BaseModel,
                 loss="bpr",
                 n_iter=100,
                 testing_epochs=5,
                 batch_size=16,
                 reg_l2=1e-3,
                 learning_rate=1e-4,
                 early_stopping=0,  # means no early stopping
                 decay_step=None,
                 decay_weight=None,
                 optimizer_func=None,
                 use_cuda=False,
                 num_negative_samples=4,
                 logfolder=None,
                 curr_date=None,
                 seed=None,
                 **kargs):
        super(CharManFitterQueryRepr1, self).__init__(net, loss, n_iter, testing_epochs, batch_size, reg_l2, learning_rate,
                                            early_stopping, decay_step, decay_weight, optimizer_func,
                                            use_cuda, num_negative_samples, logfolder, curr_date, seed, **kargs)
        self.output_size = kargs["output_size"]

    def fit(self,
            train_iteractions: interactions.ClassificationInteractions,
            verbose=True,  # for printing out evaluation during training
            topN=10,
            val_interactions: interactions.ClassificationInteractions = None,
            test_interactions: interactions.ClassificationInteractions = None):
        """
        Fit the model.
        Parameters
        ----------
        train_iteractions: :class:`interactions.ClassificationInteractions` The input sequence dataset.
        val_interactions: :class:`interactions.ClassificationInteractions`
        test_interactions: :class:`interactions.ClassificationInteractions`
        """
        self._initialize(train_iteractions)
        best_val_f1_macro, best_epoch = 0, 0
        test_results_dict = None
        iteration_counter = 0
        count_patience_epochs = 0

        for epoch_num in range(self._n_iter):

            # ------ Move to here ----------------------------------- #
            self._net.train(True)
            query_ids, left_contents, left_lengths, query_sources, query_char_sources, query_adj, \
            evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources, evd_cnt_each_query, evd_char_sources, \
            pair_labels, evd_docs_adj = self._sampler.get_train_instances_char_man(train_iteractions,
                                                                                   self.fixed_num_evidences)

            queries, query_content, query_lengths, query_sources, query_char_sources, query_adj, \
            evd_docs, evd_docs_contents, evd_docs_lens, evd_sources, evd_cnt_each_query, evd_char_sources, \
            pair_labels, evd_docs_adj = my_utils.shuffle(query_ids, left_contents, left_lengths, query_sources,
                                                         query_char_sources, query_adj, evd_docs_ids, evd_docs_contents,
                                                         evd_docs_lens, evd_sources, evd_cnt_each_query,
                                                         evd_char_sources, pair_labels, evd_docs_adj)
            epoch_loss, total_pairs = 0.0, 0
            t1 = time.time()
            for (minibatch_num,
                 (batch_query, batch_query_content, batch_query_len, batch_query_sources, batch_query_chr_src,
                  batch_query_adj, batch_evd_docs, batch_evd_contents, batch_evd_lens, batch_evd_sources,
                  # i.e. claim source
                  batch_evd_cnt_each_query, batch_evd_chr_src, batch_labels, batch_evd_docs_adj)) \
                    in enumerate(my_utils.minibatch(queries, query_content, query_lengths, query_sources,
                                                    query_char_sources, query_adj,
                                                    evd_docs, evd_docs_contents, evd_docs_lens, evd_sources,
                                                    evd_cnt_each_query, evd_char_sources, pair_labels, evd_docs_adj,
                                                    batch_size=self._batch_size)):

                batch_query = my_utils.gpu(torch.from_numpy(batch_query), self._use_cuda)
                batch_query_content = my_utils.gpu(torch.from_numpy(batch_query_content), self._use_cuda)
                # batch_query_len = my_utils.gpu(torch.from_numpy(batch_query_len), self._use_cuda)
                batch_query_sources = my_utils.gpu(torch.from_numpy(batch_query_sources), self._use_cuda)
                batch_query_chr_src = my_utils.gpu(torch.from_numpy(batch_query_chr_src), self._use_cuda)
                batch_query_adj = my_utils.gpu(torch.from_numpy(batch_query_adj), self._use_cuda)

                batch_evd_docs = my_utils.gpu(torch.from_numpy(batch_evd_docs), self._use_cuda)
                batch_evd_contents = my_utils.gpu(torch.from_numpy(batch_evd_contents), self._use_cuda)
                # batch_evd_lens = my_utils.gpu(torch.from_numpy(batch_evd_lens), self._use_cuda)
                batch_evd_sources = my_utils.gpu(torch.from_numpy(batch_evd_sources), self._use_cuda)
                batch_evd_cnt_each_query = my_utils.gpu(torch.from_numpy(batch_evd_cnt_each_query), self._use_cuda)
                batch_evd_chr_src = my_utils.gpu(torch.from_numpy(batch_evd_chr_src), self._use_cuda)

                batch_labels = my_utils.gpu(torch.from_numpy(batch_labels), self._use_cuda)
                batch_evd_docs_adj = my_utils.gpu(torch.from_numpy(batch_evd_docs_adj), self._use_cuda)
                # total_pairs += self._batch_size * self.
                additional_data = {KeyWordSettings.EvidenceCountPerQuery: batch_evd_cnt_each_query,
                                   KeyWordSettings.FCClass.QueryCharSource: batch_query_chr_src,
                                   KeyWordSettings.FCClass.DocCharSource: batch_evd_chr_src,
                                   KeyWordSettings.Query_Adj: batch_query_adj,
                                   KeyWordSettings.Evd_Docs_Adj: batch_evd_docs_adj}
                self._optimizer.zero_grad()
                if self._loss in ["bpr", "hinge", "pce", "bce", "cross_entropy",
                                  "vanilla_cross_entropy", "regression_loss", "masked_cross_entropy"]:
                    loss = self._get_multiple_evidences_predictions_normal(
                        batch_query, batch_query_content, batch_query_len, batch_query_sources,
                        batch_evd_docs, batch_evd_contents, batch_evd_lens, batch_evd_sources,
                        batch_labels, self.fixed_num_evidences, **additional_data)

                # print("Loss: ", loss)
                epoch_loss += loss.item()
                iteration_counter += 1
                # if iteration_counter % 2 == 0: break
                TensorboardWrapper.mywriter().add_scalar("loss/minibatch_loss", loss.item(), iteration_counter)
                loss.backward()
                self._optimizer.step()
                # for name, param in self._net.named_parameters():
                #     self.tensorboard_writer.add_histogram(name + "/grad", param.grad, iteration_counter)
                #     self.tensorboard_writer.add_histogram(name + "/value", param, iteration_counter)
            # epoch_loss /= float(total_pairs)
            TensorboardWrapper.mywriter().add_scalar("loss/epoch_loss_avg", epoch_loss, epoch_num)
            # print("Number of Minibatches: ", minibatch_num, "Avg. loss of epoch: ", epoch_loss)
            t2 = time.time()
            epoch_train_time = t2 - t1
            if verbose:  # validation after each epoch
                f1_macro_val = self._output_results_every_epoch(topN, val_interactions, test_interactions,
                                                                         epoch_num, epoch_train_time, epoch_loss)
                if f1_macro_val > best_val_f1_macro :
                    # if (hits + ndcg) > (best_hit + best_ndcg):
                    count_patience_epochs = 0
                    with open(self.saved_model, "wb") as f:
                        torch.save(self._net.state_dict(), f)
                    # test_results_dict = result_test
                    best_val_f1_macro, best_epoch = f1_macro_val, epoch_num
                    # test_hit, test_ndcg = hits_test, ndcg_test
                else:
                    count_patience_epochs += 1
                if self._early_stopping_patience and count_patience_epochs > self._early_stopping_patience:
                    self.output_handler.myprint(
                        "Early Stopped due to no better performance in %s epochs" % count_patience_epochs)
                    break

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'.format(epoch_loss))
        self._flush_training_results(best_val_f1_macro, best_epoch)

    def _flush_training_results(self, best_val_f1_macro: float, best_epoch: int):
        self.output_handler.myprint("Closing tensorboard")
        TensorboardWrapper.mywriter().close()
        self.output_handler.myprint('Best result: | vad F1_macro = %.5f | epoch = %d' % (best_val_f1_macro, best_epoch))

    def _get_multiple_evidences_predictions_normal(self, query_ids: torch.Tensor,
                                                   query_contents: torch.Tensor,
                                                   query_lens: np.ndarray,
                                                   query_sources: torch.Tensor,
                                                   evd_doc_ids: torch.Tensor,
                                                   evd_doc_contents: torch.Tensor,
                                                   evd_docs_lens: np.ndarray,
                                                   evd_sources: torch.Tensor,
                                                   labels: np.ndarray,
                                                   n: int, **kargs) -> torch.Tensor:
        """
        compute cross entropy loss
        Parameters
        ----------
        query_ids: (B, )
        query_contents: (B, L)
        query_lens: (B, )
        evd_doc_ids: (B, n)
        evd_doc_contents: (B, n, R)
        evd_docs_lens: (B, n)
        evd_sources: (B, n)
        labels: (B, ) labels of pair
        n: `int` is the number of evidences for each claim/query
        kargs: `dict` include: query_adj: (B,L,L), evd_docs_adj: (B, n, R, R)
        Returns
        -------
            loss value based on a loss function
        """
        evd_count_per_query = kargs[KeyWordSettings.EvidenceCountPerQuery]  # (B, )
        query_char_source = kargs[KeyWordSettings.FCClass.QueryCharSource]
        doc_char_source = kargs[KeyWordSettings.FCClass.DocCharSource]
        query_adj = kargs[KeyWordSettings.Query_Adj]
        evd_docs_adj = kargs[KeyWordSettings.Evd_Docs_Adj]
        assert evd_doc_ids.size() == evd_docs_lens.shape
        assert query_ids.size(0) == evd_doc_ids.size(0)
        assert query_lens.shape == labels.size()
        assert query_contents.size(0) == evd_doc_contents.size(0)  # = batch_size
        _, L = query_contents.size()
        batch_size = query_ids.size(0)
        # prunning at this step to remove padding\
        e_lens, e_conts, q_conts, q_lens, e_adj = [], [], [], [], []
        e_chr_src_conts = []
        expaned_labels = []
        for evd_cnt, q_cont, q_len, evd_lens, evd_doc_cont, evd_chr_src, label, evd_adj in \
                zip(evd_count_per_query, query_contents, query_lens,
                    evd_docs_lens, evd_doc_contents, doc_char_source, labels, evd_docs_adj):
            evd_cnt = int(torch_utils.cpu(evd_cnt).detach().numpy())
            e_lens.extend(list(evd_lens[:evd_cnt]))
            e_conts.append(evd_doc_cont[:evd_cnt, :])  # stacking later
            e_adj.append(evd_adj[:evd_cnt])
            e_chr_src_conts.append(evd_chr_src[:evd_cnt, :])
            q_lens.extend([q_len] * evd_cnt)
            q_conts.append(q_cont.unsqueeze(0).expand(evd_cnt, L))
            expaned_labels.extend([int(torch_utils.cpu(label).detach().numpy())] * evd_cnt)
        # concat
        e_conts = torch.cat(e_conts, dim=0)  # (n1 + n2 + ..., R)
        e_chr_src_conts = torch.cat(e_chr_src_conts, dim=0)  # (n1 + n2 + ... , R)
        e_adj = torch.cat(e_adj, dim=0)     # (n1 + n2 + ..., R, R)
        e_lens = np.array(e_lens)  # (n1 + n2 + ..., )
        q_conts = torch.cat(q_conts, dim=0)  # (n1 + n2 + ..., R)
        q_lens = np.array(q_lens)
        assert q_conts.size(0) == q_lens.shape[0] == e_conts.size(0) == e_lens.shape[0]

        d_new_indices, d_old_indices = torch_utils.get_sorted_index_and_reverse_index(e_lens)
        e_lens = my_utils.gpu(torch.from_numpy(e_lens), self._use_cuda)
        x = query_lens
        q_new_indices, q_restoring_indices = torch_utils.get_sorted_index_and_reverse_index(x)
        x = my_utils.gpu(torch.from_numpy(x), self._use_cuda)
        # query_lens = my_utils.gpu(torch.from_numpy(query_lens), self._use_cuda)

        additional_paramters = {
            KeyWordSettings.Query_lens: x,  # 每一个query长度
            KeyWordSettings.Doc_lens: evd_docs_lens,
            KeyWordSettings.DocLensIndices: (d_new_indices, d_old_indices, e_lens),
            KeyWordSettings.QueryLensIndices: (q_new_indices, q_restoring_indices, x),
            KeyWordSettings.QuerySources: query_sources,
            KeyWordSettings.DocSources: evd_sources,
            KeyWordSettings.TempLabel: labels,
            KeyWordSettings.DocContentNoPaddingEvidence: e_conts,
            KeyWordSettings.QueryContentNoPaddingEvidence: q_conts,
            KeyWordSettings.EvidenceCountPerQuery: evd_count_per_query,
            KeyWordSettings.FCClass.QueryCharSource: query_char_source,  # (B, 1, L)
            KeyWordSettings.FCClass.DocCharSource: e_chr_src_conts,
            KeyWordSettings.FIXED_NUM_EVIDENCES: n,
            KeyWordSettings.Query_Adj: query_adj,
            KeyWordSettings.Evd_Docs_Adj: e_adj                       # flatten->(n1 + n2 ..., R, R)
        }

        # (B,)
        predictions = self._net(query_contents, evd_doc_contents, **additional_paramters)
        # labels.unsqueeze(-1).expand(batch_size, n).reshape(batch_size * n)
        # labels = torch_utils.gpu(torch.from_numpy(np.array(expaned_labels)), self._use_cuda)
        # print("Labels: ", labels)
        # mask = (evd_doc_ids >= 0).view(batch_size * n).float()
        return self._loss_func(predictions, labels.float())

    def evaluate(self, testRatings: interactions.ClassificationInteractions, K: int, output_ranking=False, **kargs):
        """
        Compute evaluation metrics. No swearing in code please!!!
        Parameters
        ----------
        testRatings
        K
        output_ranking: whether we should output predictions
        kargs

        Returns
        -------

        """
        all_labels = []
        all_final_preds = []
        all_final_probs = []
        list_error_analysis = []
        for query, evidences_info in testRatings.dict_claims_and_evidences_test.items():
            evd_ids, labels, evd_contents, evd_lengths, evd_adj = evidences_info
            assert len(set(labels)) == 1, "Must have only one label due to same claim"
            all_labels.append(labels[0])
            claim_content = testRatings.dict_claim_contents[query]
            claim_source = np.array([testRatings.dict_claim_source[query]])  # (1, )
            claim_char_src = np.array([testRatings.dict_char_left_src[query]])
            evd_sources = np.array([testRatings.dict_evd_source[e] for e in evd_ids])  # (len(labels), 1)
            evd_sources = self._pad_article_sources(evd_sources)  # (1, 30)
            evd_char_src = np.array([testRatings.dict_char_right_src[e] for e in evd_ids])  # (len(labels), 1)
            query_len = np.array([testRatings.dict_claim_lengths[query]])  # shape = (1, ) where B =1
            # doc_lens = [testRatings.dict_doc_lengths[d] for d in docs]

            claim_content = np.tile(claim_content, (1, 1))  # (1, L)
            L = claim_content.shape[1]
            evd_contents = np.array(evd_contents).reshape(1, len(labels), -1)  # shape = (1, n, R)
            padded_evd_contents = self._pad_evidences(evd_contents)
            # claim_content = my_utils.gpu(claim_content)
            # evd_contents = my_utils.gpu(evd_contents)

            claim_content = my_utils.gpu(my_utils.numpy2tensor(claim_content, dtype=torch.int), self._use_cuda)
            evd_contents = my_utils.gpu(my_utils.numpy2tensor(evd_contents, dtype=torch.int),
                                        self._use_cuda)  # (1, x, R)
            padded_evd_contents = my_utils.gpu(my_utils.numpy2tensor(padded_evd_contents, dtype=torch.int),
                                               self._use_cuda)  # (1, x, R)

            # for evidences
            evd_lengths = np.array(evd_lengths)
            d_new_indices, d_old_indices = torch_utils.get_sorted_index_and_reverse_index(evd_lengths)
            evd_lengths = my_utils.gpu(my_utils.numpy2tensor(evd_lengths, dtype=torch.int), self._use_cuda)
            x = query_len  # np.repeat(query_len, len(labels))
            q_new_indices, q_restoring_indices = torch_utils.get_sorted_index_and_reverse_index(x)
            x = my_utils.gpu(my_utils.numpy2tensor(x, dtype=torch.int), self._use_cuda)

            # for sources
            claim_source = my_utils.gpu(my_utils.numpy2tensor(claim_source, dtype=torch.int), self._use_cuda)
            evd_sources = my_utils.gpu(my_utils.numpy2tensor(evd_sources, dtype=torch.int), self._use_cuda)
            claim_char_src = my_utils.gpu(my_utils.numpy2tensor(claim_char_src, dtype=torch.int), self._use_cuda)
            evd_char_src = my_utils.gpu(my_utils.numpy2tensor(evd_char_src, dtype=torch.int), self._use_cuda)

            # for query/evdience adj
            query_adj = np.array([testRatings.dict_query_adj[query]])           # (1, L, L)
            query_adj = my_utils.gpu(torch.from_numpy(query_adj), self._use_cuda)

            # query_adj = my_utils.gpu(my_utils.numpy2tensor(query_adj, dtype=torch.int), self._use_cuda)

            evd_adj = np.array([testRatings.dict_doc_adj[e] for e in evd_ids])               # (n, R, R)
            evd_adj =  my_utils.gpu(torch.from_numpy(evd_adj), self._use_cuda)

            # evd_adj =  my_utils.gpu(my_utils.numpy2tensor(evd_adj, dtype=torch.int), self._use_cuda)

            additional_information = {
                KeyWordSettings.Query_lens: x,
                KeyWordSettings.QueryLensIndices: (q_new_indices, q_restoring_indices, x),
                KeyWordSettings.Doc_lens: evd_lengths,
                KeyWordSettings.DocLensIndices: (d_new_indices, d_old_indices, evd_lengths),
                KeyWordSettings.QuerySources: claim_source,
                KeyWordSettings.DocSources: evd_sources,  # (B = 1, n = 30)
                KeyWordSettings.DocContentNoPaddingEvidence: evd_contents.view(1 * len(labels), -1),  # (B1, R)
                KeyWordSettings.FIXED_NUM_EVIDENCES: self.fixed_num_evidences,
                KeyWordSettings.EvidenceCountPerQuery: torch_utils.gpu(torch.from_numpy(np.array([len(labels)])),
                                                                       self._use_cuda),
                KeyWordSettings.QueryContentNoPaddingEvidence: claim_content.expand(len(labels), L),
                KeyWordSettings.OutputRankingKey: output_ranking,
                KeyWordSettings.FCClass.QueryCharSource: claim_char_src.long(),
                KeyWordSettings.FCClass.DocCharSource: evd_char_src.long(),
                KeyWordSettings.Query_Adj: query_adj,
                KeyWordSettings.Evd_Docs_Adj: evd_adj
            }

            # padded_evd_contents = self._pad_evidences(evd_contents) # 1, 30, R
            probs = self._net.predict(claim_content, padded_evd_contents,
                                      **additional_information)  # shape = (len(labels), )
            if output_ranking:
                probs, att_scores = probs
                predictions = probs.argmax(dim=1)
                more_info = {KeyWordSettings.FCClass.AttentionWeightsInfo: att_scores}
                list_error_analysis.append(
                    self._prepare_error_analysis(testRatings, query, evd_ids, probs, labels, **more_info))
            else:
                predictions = probs.argmax(dim=1)
            all_final_preds.append(float(my_utils.cpu(predictions).detach().numpy().flatten()))
            all_final_probs.append(float(my_utils.cpu(probs[:, 1]).detach().numpy().flatten()))

        results = self._computing_metrics(true_labels=all_labels, predicted_labels=all_final_preds, predicted_probs=all_final_probs)
        if output_ranking: return results, list_error_analysis  # sorted(list_error_analysis, key=lambda x: x["qid"])
        return results

    def _computing_metrics(self, true_labels: List[int], predicted_labels: List[float], predicted_probs: List[float]):
        """
        Computing classifiction metrics for 3 category classification
        Parameters
        ----------
        true_labels: ground truth
        predicted_labels: predicted labels

        Returns
        -------

        """
        assert len(true_labels) == len(predicted_labels)
        results = {}

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_labels, predicted_probs, pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)
        f1_macro = f1_score(true_labels, predicted_labels, average='macro')
        f1_micro = f1_score(true_labels, predicted_labels, average='micro')
        f1 = f1_score(true_labels, predicted_labels)

        # this is the normal precision and recall we seen so many times
        precision_true_class = precision_score(true_labels, predicted_labels, labels=[1], average=None)[0]
        recall_true_class = recall_score(true_labels, predicted_labels, labels=[1], average=None)[0]
        f1_true_class = f1_score(true_labels, predicted_labels, labels=[1], average=None)[0]

        precision_false_class = precision_score(true_labels, predicted_labels, labels=[0], average=None)[0]
        recall_false_class = recall_score(true_labels, predicted_labels, labels=[0], average=None)[0]
        f1_false_class = f1_score(true_labels, predicted_labels, labels=[0], average=None)[0]

        precision_mixed_class = precision_score(true_labels, predicted_labels, labels=[2], average=None)[0] \
            if self.output_size == 3 else 0.0
        recall_mixed_class = recall_score(true_labels, predicted_labels, labels=[2], average=None)[0] \
            if self.output_size == 3 else 0.0
        f1_mixed_class = f1_score(true_labels, predicted_labels, labels=[2], average=None)[0] \
            if self.output_size == 3 else 0.0

        results[KeyWordSettings.AUC_metric] = auc
        results[KeyWordSettings.F1_macro] = f1_macro
        results[KeyWordSettings.F1_micro] = f1_micro
        results[KeyWordSettings.F1] = f1

        results[KeyWordSettings.PrecisionTrueCls] = precision_true_class
        results[KeyWordSettings.RecallTrueCls] = recall_true_class
        results[KeyWordSettings.F1TrueCls] = f1_true_class  # this must be normal F1

        results[KeyWordSettings.PrecisionFalseCls] = precision_false_class
        results[KeyWordSettings.RecallFalseCls] = recall_false_class
        results[KeyWordSettings.F1FalseCls] = f1_false_class

        results[KeyWordSettings.PrecisionMixedCls] = precision_mixed_class
        results[KeyWordSettings.RecallMixedCls] = recall_mixed_class
        results[KeyWordSettings.F1MixedCls] = f1_mixed_class

        return results

    def _prepare_error_analysis(self, testRatings: interactions.ClassificationInteractions,
                                query: int, evd_ids: List[int], probs: float, labels: List[int], **kargs):
        word_att_weights, evd_att_weight = kargs[
            KeyWordSettings.FCClass.AttentionWeightsInfo]  # () vs. (1, 30, num_heads)
        # for doc attention weight
        evd_att_weight = evd_att_weight.squeeze(0)  # .squeeze(-1)
        evd_att_weight = torch_utils.cpu(evd_att_weight).detach().numpy()
        evd_att_weight = evd_att_weight[:len(evd_ids)]  # take only len(evd_ids)

        # indices = np.argsort(-evd_att_weight, axis=0)  # (x, num_head) we sort on the first dimension of each head

        assert np.sum(abs(np.sum(evd_att_weight, axis=0) - 1.0)) < 1e-5, np.sum(
            abs(np.sum(evd_att_weight, axis=0) - 1.0))
        # for word attention
        # word_att_weights = word_att_weights  # .squeeze(-1)  # (n <= 30, R, 1) -> (n, R) where n = len(evd_ids)
        word_att_weights = torch_utils.cpu(word_att_weights).detach().numpy()
        # swapping indices to push most important document up
        # evd_ids = np.array(evd_ids)[indices]
        # evd_att_weight = evd_att_weight[indices]
        # word_att_weights = word_att_weights[indices]
        # done swapping
        ranked_doc_list = []
        for d, doc_score, word_scores in zip(evd_ids, evd_att_weight, word_att_weights):
            word_att_score_str = []
            for head in range(word_scores.shape[1]):
                x = word_scores[:, head]
                assert abs(np.sum(x) - 1.0) < 1e-5, abs(np.sum(x) - 1.0)
                word_att_score_str.append(str(x.tolist()))
            # word_att_score_str = [word_scores[:, head] for head in range(word_scores.shape[1])]
            each_evd = {
                KeyWordSettings.Doc_cID: int(d),
                KeyWordSettings.Doc_wContent: testRatings.dict_doc_raw_contents[d],
                KeyWordSettings.FCClass.DocAttentionScore: str(doc_score.tolist()),
                KeyWordSettings.FCClass.WordAttentionScore: word_att_score_str,
                KeyWordSettings.DocSources: testRatings.dict_raw_evd_source[d]
            }
            ranked_doc_list.append(each_evd)
        # ranked_doc_list = [{KeyWordSettings.Doc_cID: int(d),
        #                     KeyWordSettings.Doc_wContent: testRatings.dict_doc_raw_contents[d],
        #                     KeyWordSettings.FCClass.DocAttentionScore: float(doc_score),
        #                     KeyWordSettings.FCClass.WordAttentionScore: [str(arr) for arr in word_scores.tolist()],
        #                     KeyWordSettings.DocSources: testRatings.dict_raw_evd_source[d]}
        #                    for d, doc_score, word_scores in zip(evd_ids, evd_att_weight, word_att_weights)]
        q_details = {KeyWordSettings.Query_id: int(query),
                     KeyWordSettings.QuerySources: testRatings.dict_raw_claim_source[int(query)],
                     KeyWordSettings.FCClass.ClaimLabel: str(labels[0]),
                     KeyWordSettings.FCClass.PredictedProb: my_utils.cpu(probs).detach().numpy().flatten().tolist(),
                     # return probs of every class
                     KeyWordSettings.Ranked_Docs: ranked_doc_list,
                     KeyWordSettings.Query_Content: testRatings.dict_query_raw_contents[query]}
        return q_details

    def _output_results_every_epoch(self, topN: int, val_interactions: interactions.ClassificationInteractions,
                                    test_interactions: interactions.ClassificationInteractions,
                                    epoch_num: int, epoch_train_time: float, epoch_loss: float):
        t1 = time.time()
        assert len(val_interactions.dict_claims_and_evidences_test) in KeyWordSettings.ClaimCountVal, \
            len(val_interactions.dict_claims_and_evidences_test)
        result_val = self.evaluate(val_interactions, topN)
        auc_val = result_val[KeyWordSettings.AUC_metric]
        f1_macro_val = result_val[KeyWordSettings.F1_macro]
        f1_micro_val = result_val[KeyWordSettings.F1_micro]
        # f1_val = result_val[KeyWordSettings.F1]
        # ndcg_val = result_val["ndcg"]
        t2 = time.time()
        valiation_time = t2 - t1

        if epoch_num and epoch_num % self._testing_epochs == 0:
            t1 = time.time()
            assert len(test_interactions.dict_claims_and_evidences_test) in KeyWordSettings.ClaimCountTest, \
                len(test_interactions.dict_claims_and_evidences_test)
            result_test = self.evaluate(test_interactions, topN)
            # auc_test = result_test[KeyWordSettings.AUC_metric]
            f1_macro_test = result_test[KeyWordSettings.F1_macro]
            f1_micro_test = result_test[KeyWordSettings.F1_micro]
            # f1_test = result_test[KeyWordSettings.F1]
            # ndcg_test = result_test["ndcg"]
            t2 = time.time()
            testing_time = t2 - t1
            # TensorboardWrapper.mywriter().add_scalar("auc/auc_test", auc_test, epoch_num)
            TensorboardWrapper.mywriter().add_scalar("f1/f1_macro_test", f1_macro_test, epoch_num)
            TensorboardWrapper.mywriter().add_scalar("f1/f1_micro_test", f1_micro_test, epoch_num)
            # TensorboardWrapper.mywriter().add_scalar("f1/f1_test", f1_test, epoch_num)

            # TensorboardWrapper.mywriter().add_scalar("ndcg/ndcg_test", ndcg_test, epoch_num)
            self.output_handler.myprint('|Epoch %03d | Test F1_macro = %.5f | Testing time: %04.1f(s)'
                                        % (epoch_num, f1_macro_test, testing_time))

        # TensorboardWrapper.mywriter().add_scalar("auc/auc_val", auc_val, epoch_num)
        TensorboardWrapper.mywriter().add_scalar("f1/f1_macro_val", f1_macro_val, epoch_num)
        TensorboardWrapper.mywriter().add_scalar("f1/f1_micro_val", f1_micro_val, epoch_num)
        # TensorboardWrapper.mywriter().add_scalar("f1/f1_val", f1_val, epoch_num)

        # TensorboardWrapper.mywriter().add_scalar("ndcg/ndcg_val", ndcg, epoch_num)
        self.output_handler.myprint('|Epoch %03d | Train time: %04.1f(s) | Train loss: %.3f'
                                    '| Val F1_macro = %.3f | Vad AUC = %.3f'
                                    '| Val F1_micro = %.3f | Validation time: %04.1f(s)'
                                    % (epoch_num, epoch_train_time, epoch_loss, f1_macro_val, auc_val,
                                       f1_micro_val, valiation_time))
        return f1_macro_val


    def load_best_model(self, val_interactions: interactions.ClassificationInteractions,
                        test_interactions: interactions.ClassificationInteractions, topN: int = 10):
        mymodel = self._net
        # print("Trained model: ", mymodel.out.weight)
        mymodel.load_state_dict(torch.load(self.saved_model))
        mymodel.train(False)
        my_utils.gpu(mymodel, self._use_cuda)

        assert len(val_interactions.dict_claims_and_evidences_test) in KeyWordSettings.ClaimCountVal
        result_val, error_analysis_val = self.evaluate(val_interactions, topN, output_ranking=True)
        auc_val = result_val[KeyWordSettings.AUC_metric]
        f1_val = result_val[KeyWordSettings.F1]
        f1_macro_val = result_val[KeyWordSettings.F1_macro]
        f1_micro_val = result_val[KeyWordSettings.F1_micro]

        assert len(test_interactions.dict_claims_and_evidences_test) in KeyWordSettings.ClaimCountTest
        result_test, error_analysis_test = self.evaluate(test_interactions, topN, output_ranking=True)
        auc_test = result_test[KeyWordSettings.AUC_metric]
        f1_test = result_test[KeyWordSettings.F1]
        f1_macro = result_test[KeyWordSettings.F1_macro]
        f1_micro = result_test[KeyWordSettings.F1_micro]
        precision_true_class = result_test[KeyWordSettings.PrecisionTrueCls]
        recall_true_class = result_test[KeyWordSettings.RecallTrueCls]
        f1_true_class = result_test[KeyWordSettings.F1TrueCls]

        precision_false_class = result_test[KeyWordSettings.PrecisionFalseCls]
        recall_false_class = result_test[KeyWordSettings.RecallFalseCls]
        f1_false_class = result_test[KeyWordSettings.F1FalseCls]

        precision_mixed_class = result_test[KeyWordSettings.PrecisionMixedCls]
        recall_mixed_class = result_test[KeyWordSettings.RecallMixedCls]
        f1_mixed_class = result_test[KeyWordSettings.F1MixedCls]

        print(auc_val, auc_test)
        self.output_handler.save_error_analysis_validation(json.dumps(error_analysis_val, sort_keys=True, indent=2))
        self.output_handler.save_error_analysis_testing(json.dumps(error_analysis_test, sort_keys=True, indent=2))
        self.output_handler.myprint('Best Vad F1_macro = %.5f | Best Vad AUC = %.5f'
                                    '| Best Test F1_macro = %.5f | Best Test F1_micro = %.5f | Best Vad AUC = %.5f \n'
                                    '| Best Test Precision_True_class = %.5f | Best Test Recall_True_class = %.5f '
                                    '| Best Test F1_True_class = %.5f \n'
                                    '| Best Test Precision_False_class = %.5f | Best Test_Recall_False class = %.5f '
                                    '| Best Test F1_False_class = %.5f \n'
                                    '| Best Test Precision_Mixed_class = %.5f | Best Test_Recall_Mixed_class = %.5f '
                                    '| Best Test F1_Mixed_class = %.5f '
                                    % (f1_macro_val, auc_val, f1_macro, f1_micro, auc_test,
                                       precision_true_class, recall_true_class, f1_true_class,
                                       precision_false_class, recall_false_class, f1_false_class,
                                       precision_mixed_class, recall_mixed_class, f1_mixed_class))

        return result_val, result_test
    # def _get_adj(self, x: torch.Tensor, lens: torch.Tensor):
    #     """
    #     x: 'torch.Tensor' of shape (B, L)
    #     """
    #     B, L = x.size()
    #     adj = np.zeros((B, L, L), np.int64)
    #     for b in range(B):
    #         l = lens[b]
    #         for i in range(l):
    #             adj[b, i, max(0, i-2):min(i+3, l)] = 1
    #             for j in range(i+3, l):
    #                 if x[b][i] == x[b][j]:
    #                     adj[b][i][j] = adj[b][j][i] = 1
    #     return my_utils.gpu(torch.from_numpy(adj), self._use_cuda)