import torch
import numpy as np
import torch_utils
from Models import base_model
import losses as my_losses
import torch_utils as my_utils
import torch.optim as optim
import time
import json
import interactions
from handlers.tensorboard_writer import TensorboardWrapper
from setting_keywords import KeyWordSettings
import sklearn
from typing import List
from Fitting.densebaseline_fit import DenseBaselineFitter
from sklearn.metrics import f1_score, precision_score, recall_score


class DeclareFitter(DenseBaselineFitter):

    def __init__(self, net: base_model.BaseModel,
                 loss = "bpr",
                 n_iter = 100,
                 testing_epochs = 5,
                 batch_size = 16,
                 reg_l2 = 1e-3,
                 learning_rate = 1e-4,
                 early_stopping = 0,  # means no early stopping
                 decay_step = None,
                 decay_weight = None,
                 optimizer_func = None,
                 use_cuda = False,
                 num_negative_samples = 4,
                 logfolder = None,
                 curr_date = None,
                 seed = None,
                 **kargs):
        super(DeclareFitter, self).__init__(net, loss, n_iter, testing_epochs, batch_size, reg_l2, learning_rate,
                                           early_stopping, decay_step, decay_weight, optimizer_func,
                                           use_cuda, num_negative_samples, logfolder, curr_date, seed, **kargs)
        self.fixed_num_evidences = kargs[KeyWordSettings.FIXED_NUM_EVIDENCES]
        self.output_handler = kargs[KeyWordSettings.OutputHandlerFactChecking]

    def _initialize(self, interactions: interactions.MatchInteraction):
        """

        Parameters
        ----------
        interactions: :class:`interactions.MatchInteraction`
        Returns
        -------

        """
        # put the model into cuda if use cuda
        self._net = my_utils.gpu(self._net, self._use_cuda)

        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay = self._reg_l2,
                lr = self._learning_rate)
        else:
            self._optimizer = self._optimizer_func(self._net.parameters())

        # losses functions
        if self._loss == "cross_entropy":
            # self._loss_func = my_losses.binary_cross_entropy_cls
            self._loss_func = my_losses.cross_entroy

        self.output_handler.myprint("Using: " + str(self._loss_func))

    def fit(self, train_iteractions: interactions.ClassificationInteractions,
            verbose = True,  # for printing out evaluation during training
            topN = 10,
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
        best_val_auc, best_val_f1_macro, best_epoch, test_auc = 0, 0, 0, 0
        test_results_dict = None
        iteration_counter = 0
        count_patience_epochs = 0

        for epoch_num in range(self._n_iter):

            # ------ Move to here ----------------------------------- #
            self._net.train(True)
            query_ids, left_contents, left_lengths, query_sources, \
            evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources, \
            pair_labels = self._sampler.get_train_instances_declare(train_iteractions, self.fixed_num_evidences)

            queries, query_content, query_lengths, query_sources_shuffled, \
            evd_docs, evd_docs_contents, evd_docs_lens, evd_sources_shuffled, \
            pair_labels_shuffled = my_utils.shuffle(query_ids, left_contents, left_lengths, query_sources,
                                                    evd_docs_ids, evd_docs_contents, evd_docs_lens, evd_sources,
                                                    pair_labels)
            epoch_loss, total_pairs = 0.0, 0
            t1 = time.time()
            for (minibatch_num,
                (batch_query, batch_query_content, batch_query_len, batch_query_sources,  # i.e. claim source
                 batch_evd_docs, batch_evd_contents, batch_evd_lens, batch_evd_sources, batch_labels)) \
                    in enumerate(my_utils.minibatch(queries, query_content, query_lengths, query_sources_shuffled,
                                                    evd_docs, evd_docs_contents, evd_docs_lens, evd_sources_shuffled,
                                                    pair_labels_shuffled, batch_size = self._batch_size)):

                batch_query = my_utils.gpu(torch.from_numpy(batch_query), self._use_cuda)
                batch_query_content = my_utils.gpu(torch.from_numpy(batch_query_content), self._use_cuda)
                # batch_query_len = my_utils.gpu(torch.from_numpy(batch_query_len), self._use_cuda)
                batch_query_sources = my_utils.gpu(torch.from_numpy(batch_query_sources), self._use_cuda)

                batch_evd_docs = my_utils.gpu(torch.from_numpy(batch_evd_docs), self._use_cuda)
                batch_evd_contents = my_utils.gpu(torch.from_numpy(batch_evd_contents), self._use_cuda)
                # batch_evd_lens = my_utils.gpu(torch.from_numpy(batch_evd_lens), self._use_cuda)
                batch_evd_sources = my_utils.gpu(torch.from_numpy(batch_evd_sources), self._use_cuda)

                batch_labels = my_utils.gpu(torch.from_numpy(batch_labels), self._use_cuda)
                # total_pairs += self._batch_size * self.

                self._optimizer.zero_grad()
                if self._loss in ["bpr", "hinge", "pce", "bce", "cross_entropy", "vanilla_cross_entropy"]:
                    loss = self._get_multiple_evidences_predictions_normal(
                        batch_query, batch_query_content, batch_query_len, batch_query_sources,
                        batch_evd_docs, batch_evd_contents, batch_evd_lens, batch_evd_sources,
                        batch_labels, self.fixed_num_evidences)

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
                f1_macro_val, auc_val = self._output_results_every_epoch(topN, val_interactions, test_interactions, epoch_num, epoch_train_time, epoch_loss)
                if (f1_macro_val > best_val_f1_macro) or \
                        (f1_macro_val == best_val_f1_macro and auc_val > best_val_auc):  # prioritize f1_macro
                    # if (hits + ndcg) > (best_hit + best_ndcg):
                    count_patience_epochs = 0
                    with open(self.saved_model, "wb") as f:
                        torch.save(self._net.state_dict(), f)
                    # test_results_dict = result_test
                    best_val_auc, best_val_f1_macro, best_epoch = auc_val, f1_macro_val, epoch_num
                    # test_hit, test_ndcg = hits_test, ndcg_test

                else: count_patience_epochs += 1
                if self._early_stopping_patience and count_patience_epochs > self._early_stopping_patience:
                    self.output_handler.myprint("Early Stopped due to no better performance in %s epochs" % count_patience_epochs)
                    break

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'.format(epoch_loss))

        self._flush_training_results(best_val_auc, best_epoch)

    def _flush_training_results(self, best_val_auc: float, best_epoch: int):
        self.output_handler.myprint("Closing tensorboard")
        TensorboardWrapper.mywriter().close()
        self.output_handler.myprint('Best result: | vad auc = %.5f | epoch = %d' % (best_val_auc, best_epoch))

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
        f1_val = result_val[KeyWordSettings.F1]
        # ndcg_val = result_val["ndcg"]
        t2 = time.time()
        valiation_time = t2 - t1

        if epoch_num and epoch_num % self._testing_epochs == 0:
            t1 = time.time()
            assert len(test_interactions.dict_claims_and_evidences_test) in KeyWordSettings.ClaimCountTest, \
                len(test_interactions.dict_claims_and_evidences_test)
            result_test = self.evaluate(test_interactions, topN)
            auc_test = result_test[KeyWordSettings.AUC_metric]
            f1_macro_test = result_test[KeyWordSettings.F1_macro]
            f1_micro_test = result_test[KeyWordSettings.F1_micro]
            f1_test = result_test[KeyWordSettings.F1]
            # ndcg_test = result_test["ndcg"]
            t2 = time.time()
            testing_time = t2 - t1
            TensorboardWrapper.mywriter().add_scalar("auc/auc_test", auc_test, epoch_num)
            TensorboardWrapper.mywriter().add_scalar("f1/f1_macro_test", f1_macro_test, epoch_num)
            TensorboardWrapper.mywriter().add_scalar("f1/f1_micro_test", f1_micro_test, epoch_num)
            TensorboardWrapper.mywriter().add_scalar("f1/f1_test", f1_test, epoch_num)

            # TensorboardWrapper.mywriter().add_scalar("ndcg/ndcg_test", ndcg_test, epoch_num)
            self.output_handler.myprint('|Epoch %03d | Test AUC = %.5f | Testing time: %04.1f(s)'
                                % (epoch_num, auc_test, testing_time))

        TensorboardWrapper.mywriter().add_scalar("auc/auc_val", auc_val, epoch_num)
        TensorboardWrapper.mywriter().add_scalar("f1/f1_macro_val", f1_macro_val, epoch_num)
        TensorboardWrapper.mywriter().add_scalar("f1/f1_micro_val", f1_micro_val, epoch_num)
        TensorboardWrapper.mywriter().add_scalar("f1/f1_val", f1_val, epoch_num)

        # TensorboardWrapper.mywriter().add_scalar("ndcg/ndcg_val", ndcg, epoch_num)
        self.output_handler.myprint('|Epoch %03d | Train time: %04.1f(s) | Train loss: %.3f'
                            '| Val F1_macro = %.3f | Vad AUC = %.5f | Val F1 = %.5f '
                            '| Val F1_micro = %.3f | Validation time: %04.1f(s)'
                            % (epoch_num, epoch_train_time, epoch_loss, f1_macro_val, auc_val,
                               f1_val, f1_micro_val, valiation_time))
        return f1_macro_val, auc_val

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
        labels: (B, ) labels of pair
        n: `int` is the number of evidences for each claim/query
        kargs: `dict`
        Returns
        -------
            loss value based on a loss function
        """
        assert query_ids.size() == evd_doc_ids.size()
        assert query_lens.shape == evd_docs_lens.shape == labels.size()
        assert query_contents.size(0) == evd_doc_contents.size(0)  # = batch_size
        batch_size = query_ids.size(0)
        d_new_indices, d_old_indices = torch_utils.get_sorted_index_and_reverse_index(evd_docs_lens)
        q_new_indices, q_old_indices = torch_utils.get_sorted_index_and_reverse_index(query_lens)
        evd_docs_lens = my_utils.gpu(torch.from_numpy(evd_docs_lens), self._use_cuda)
        query_lens = my_utils.gpu(torch.from_numpy(query_lens), self._use_cuda)
        additional_paramters = {
            KeyWordSettings.QueryIDs: query_ids,
            KeyWordSettings.DocIDs: evd_doc_ids,
            KeyWordSettings.Query_lens: query_lens,
            KeyWordSettings.Doc_lens: evd_docs_lens,
            KeyWordSettings.DocLensIndices: (d_new_indices, d_old_indices, evd_docs_lens),
            KeyWordSettings.QueryLensIndices: (q_new_indices, q_old_indices, query_lens),
            KeyWordSettings.QuerySources: query_sources,
            KeyWordSettings.DocSources: evd_sources
        }
        predictions = self._net(query_contents, evd_doc_contents, **additional_paramters)  # (B * n, 2)
        return self._loss_func(predictions, labels)

    def load_best_model(self, val_interactions: interactions.ClassificationInteractions,
                        test_interactions: interactions.ClassificationInteractions, topN: int = 10):
        mymodel = self._net
        # print("Trained model: ", mymodel.out.weight)
        mymodel.load_state_dict(torch.load(self.saved_model))
        mymodel.train(False)
        my_utils.gpu(mymodel, self._use_cuda)

        assert len(val_interactions.dict_claims_and_evidences_test) in KeyWordSettings.ClaimCountVal
        result_val, error_analysis_val = self.evaluate(val_interactions, topN, output_ranking = True)
        auc_val = result_val[KeyWordSettings.AUC_metric]
        f1_val = result_val[KeyWordSettings.F1]

        assert len(test_interactions.dict_claims_and_evidences_test) in KeyWordSettings.ClaimCountTest
        result_test, error_analysis_test = self.evaluate(test_interactions, topN, output_ranking = True)
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

        self.output_handler.save_error_analysis_validation(json.dumps(error_analysis_val, sort_keys = True, indent = 2))
        self.output_handler.save_error_analysis_testing(json.dumps(error_analysis_test, sort_keys = True, indent = 2))
        self.output_handler.myprint('Best Vad AUC = %.5f | Best Vad F1 = %.5f '
                            '| Best Test AUC = %.5f |Best Test F1 = %.5f'
                            '| Best Test F1_macro = %.5f | Best Test F1_micro = %.5f \n'
                            '| Best Test Precision_True_class = %.5f | Best Test Recall_True_class = %.5f '
                            '| Best Test F1_True_class = %.5f \n'
                            '| Best Test Precision_False_class = %.5f | Best Test_Recall_False class = %.5f '
                            '| Best Test F1_False_class = %.5f '
                            % (auc_val, f1_val, auc_test, f1_test, f1_macro, f1_micro,
                               precision_true_class, recall_true_class, f1_true_class,
                               precision_false_class, recall_false_class, f1_false_class))

        return result_val, result_test

    def evaluate(self, testRatings: interactions.ClassificationInteractions, K: int, output_ranking = False, **kargs):
        """
        This is the fucking place where I need to compute the fucking Accuracy and other fucking shit.
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
        all_final_probs = []
        list_error_analysis = []
        for query, evidences_info in testRatings.dict_claims_and_evidences_test.items():
            evd_ids, labels, evd_contents, evd_lengths = evidences_info
            assert len(set(labels)) == 1, "Must have only one label due to same claim"
            all_labels.append(labels[0])
            claim_content = testRatings.dict_claim_contents[query]
            claim_source = np.array([testRatings.dict_claim_source[query]] * len(labels))  # (len(labels), 1)
            evd_sources = np.array([testRatings.dict_evd_source[e] for e in evd_ids])  # (len(labels), 1)
            query_len = np.array([testRatings.dict_claim_lengths[query]] * len(labels))

            # doc_lens = [testRatings.dict_doc_lengths[d] for d in docs]

            claim_content = np.tile(claim_content, (len(labels), 1))  # len(labels), query_contnt_leng)
            evd_contents = np.array(evd_contents)  # shape = (real_num_evd, R) or = (len(labels), R)
            # claim_content = my_utils.gpu(claim_content)
            # evd_contents = my_utils.gpu(evd_contents)

            claim_content = my_utils.gpu(my_utils.numpy2tensor(claim_content, dtype=torch.int), self._use_cuda)
            evd_contents = my_utils.gpu(my_utils.numpy2tensor(evd_contents, dtype=torch.int), self._use_cuda)
            # for evidences
            evd_lengths = np.array(evd_lengths)
            d_new_indices, d_old_indices = torch_utils.get_sorted_index_and_reverse_index(evd_lengths)
            q_new_indices, q_old_indices = torch_utils.get_sorted_index_and_reverse_index(query_len)
            evd_lengths = my_utils.gpu(my_utils.numpy2tensor(evd_lengths, dtype=torch.int), self._use_cuda)
            query_len = my_utils.gpu(my_utils.numpy2tensor(query_len, dtype=torch.int), self._use_cuda)

            # for sources
            claim_source = my_utils.gpu(my_utils.numpy2tensor(claim_source, dtype=torch.int), self._use_cuda)
            evd_sources = my_utils.gpu(my_utils.numpy2tensor(evd_sources, dtype=torch.int), self._use_cuda)

            query_id_tsr = torch_utils.gpu(torch.from_numpy(np.array([query])), self._use_cuda)
            additional_information = {
                KeyWordSettings.Query_lens: query_len,
                KeyWordSettings.Doc_lens: evd_lengths,
                KeyWordSettings.DocLensIndices: (d_new_indices, d_old_indices, evd_lengths),
                KeyWordSettings.QueryLensIndices: (q_new_indices, q_old_indices, query_len),
                KeyWordSettings.QuerySources: claim_source,
                KeyWordSettings.DocSources: evd_sources,
                KeyWordSettings.QueryIDs: query_id_tsr,
                KeyWordSettings.DocIDs: torch_utils.gpu(torch.from_numpy(np.array(evd_ids)), self._use_cuda)
            }
            probs = self._net.predict(claim_content, evd_contents, **additional_information)  # shape = (len(labels), )
            all_final_probs.append(float(my_utils.cpu(probs).detach().numpy().flatten()))


        results = self._computing_metrics(true_labels = all_labels, predicted_probs = all_final_probs)
        if output_ranking: return results, []  # sorted(list_error_analysis, key=lambda x: x["qid"])

        return results

    def _computing_metrics(self, true_labels: List[int], predicted_probs: List[float]):
        """
        Computing classifiction metrics for binary classification
        Parameters
        ----------
        true_labels: ground truth
        predicted_probs: predicted probabilities for true class

        Returns
        -------

        """
        assert len(true_labels) == len(predicted_probs)
        results = {}
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_labels, predicted_probs, pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)
        predicted_labels = (np.array(predicted_probs) >= 0.5) * 1
        f1_macro = f1_score(true_labels, predicted_labels, average='macro')
        f1_micro = f1_score(true_labels, predicted_labels, average='micro')
        f1 = f1_score(true_labels, predicted_labels)
        # this is the normal precision and recall we seen so many times
        precision_true_class = precision_score(true_labels, predicted_labels, pos_label = 1)
        recall_true_class = recall_score(true_labels, predicted_labels, pos_label = 1)
        f1_true_class = f1_score(true_labels, predicted_labels, pos_label = 1)

        precision_false_class = precision_score(true_labels, predicted_labels, pos_label = 0)
        recall_false_class = recall_score(true_labels, predicted_labels, pos_label = 0)
        f1_false_class = f1_score(true_labels, predicted_labels, pos_label = 0)

        results[KeyWordSettings.AUC_metric] = auc
        results[KeyWordSettings.F1_macro] = f1_macro
        results[KeyWordSettings.F1_micro] = f1_micro
        results[KeyWordSettings.F1] = f1

        results[KeyWordSettings.PrecisionTrueCls] = precision_true_class
        results[KeyWordSettings.RecallTrueCls] = recall_true_class
        results[KeyWordSettings.F1TrueCls] = f1_true_class  # this must be normal F1
        assert abs(f1_true_class - f1) < 1e-10

        results[KeyWordSettings.PrecisionFalseCls] = precision_false_class
        results[KeyWordSettings.RecallFalseCls] = recall_false_class
        results[KeyWordSettings.F1FalseCls] = f1_false_class

        return results
