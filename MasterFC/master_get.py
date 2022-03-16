import sys
sys.path.insert(0, '../../GET')
sys.path.insert(0, '../GET')

from Models.FCWithEvidences import graph_based_semantic_structure
from Fitting.FittingFC import char_man_fitter_query_repr1
import time
import json
from interactions import ClassificationInteractions
import matchzoo as mz
from handlers import cls_load_data
import argparse
import random
import numpy as np
import torch
import os
import datetime
from handlers.output_handler_FC import FileHandlerFC
from Evaluation import mzEvaluator as evaluator
from setting_keywords import KeyWordSettings
from matchzoo.embedding import entity_embedding


def fit_models(args):
    if not os.path.exists(args.log):
        os.mkdir(args.log)

    curr_date = datetime.datetime.now().timestamp()  # seconds
    # folder to store all outputed files of a run
    secondary_log_folder = os.path.join(args.log, "log_results_%s" % (args.dataset))
    if not os.path.exists(secondary_log_folder):
        os.mkdir(secondary_log_folder)
    args.secondary_log_folder = secondary_log_folder
    # args.seed = random.randint(1, 150000)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    root = os.path.join(os.path.join(args.path, args.dataset), "mapped_data")
    tx = time.time()
    kfold_dev_results, kfold_test_results = [], []
    list_metrics = KeyWordSettings.CLS_METRICS                  # evaluation metrics
    for i in range(args.num_folds):
        outfolder_per_fold = os.path.join(secondary_log_folder, "Fold_%s" % i)
        if not os.path.exists(outfolder_per_fold):
            os.mkdir(outfolder_per_fold)

        logfolder_result_per_fold = os.path.join(outfolder_per_fold, "result_%s.txt" % int(seed))
        file_handler = FileHandlerFC()
        file_handler.init_log_files(logfolder_result_per_fold)
        settings = json.dumps(vars(args), sort_keys=True, indent=2)
        file_handler.myprint("============= FOLD %s ========================" % i)
        file_handler.myprint(settings)
        file_handler.myprint("Setting seed to " + str(args.seed))
       
        predict_pack = cls_load_data.load_data(root + "/%sfold" % args.num_folds, 'test_%s' % i, kfolds = args.num_folds)
        train_pack = cls_load_data.load_data(root + "/%sfold" % args.num_folds, 'train_%s' % i, kfolds = args.num_folds)
        valid_pack = cls_load_data.load_data(root, 'dev', kfolds = args.num_folds)
        # print(train_pack.left.head())

        a = train_pack.left["text_left"].str.lower().str.split().apply(len).max()
        b = valid_pack.left["text_left"].str.lower().str.split().apply(len).max()
        c = predict_pack.left["text_left"].str.lower().str.split().apply(len).max()
        max_query_length = max([a, b, c])
        min_query_length = min([a, b, c])

        a = train_pack.right["text_right"].str.lower().str.split().apply(len).max()
        b = valid_pack.right["text_right"].str.lower().str.split().apply(len).max()
        c = predict_pack.right["text_right"].str.lower().str.split().apply(len).max()
        max_doc_length = max([a, b, c])
        min_doc_length = min([a, b, c])

        file_handler.myprint("Min query length, " + str(min_query_length) + " Min doc length " + str(min_doc_length))
        file_handler.myprint("Max query length, " + str(max_query_length) + " Max doc length " + str(max_doc_length))
        additional_data = {KeyWordSettings.OutputHandlerFactChecking: file_handler,
                           KeyWordSettings.GNN_Window: args.gnn_window_size}
        preprocessor = mz.preprocessors.CharManPreprocessor(fixed_length_left = args.fixed_length_left,
                                                            fixed_length_right = args.fixed_length_right,
                                                            fixed_length_left_src = args.fixed_length_left_src_chars,
                                                            fixed_length_right_src = args.fixed_length_right_src_chars)
        t1 = time.time()
        print('parsing data')
        train_processed = preprocessor.fit_transform(train_pack)  # This is a DataPack
        valid_processed = preprocessor.transform(valid_pack)
        predict_processed = preprocessor.transform(predict_pack)
        # print(train_processed.left.head())

        train_interactions = ClassificationInteractions(train_processed, **additional_data)
        valid_interactions = ClassificationInteractions(valid_processed, **additional_data)
        test_interactions = ClassificationInteractions(predict_processed, **additional_data)

        file_handler.myprint('done extracting')
        t2 = time.time()
        file_handler.myprint('loading data time: %d (seconds)' % (t2 - t1))
        file_handler.myprint("Building model")

        print("Loading word embeddings......")
        t1_emb = time.time()
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        glove_embedding = mz.datasets.embeddings.load_glove_embedding_FC(dimension = args.word_embedding_size,
                                                                         term_index = term_index, **additional_data)

        embedding_matrix = glove_embedding.build_matrix(term_index)
        entity_embs1 = entity_embedding.EntityEmbedding(args.claim_src_emb_size)
        claim_src_embs_matrix = entity_embs1.build_matrix(preprocessor.context['claim_source_unit'].state['term_index'])

        entity_embs2 = entity_embedding.EntityEmbedding(args.article_src_emb_size)
        article_src_embs_matrix = entity_embs2.build_matrix(preprocessor.context['article_source_unit'].state['term_index'])

        t2_emb = time.time()
        print("Time to load word embeddings......", (t2_emb - t1_emb))

        match_params = {}
        match_params['embedding'] = embedding_matrix
        match_params["num_classes"] = args.num_classes
        match_params["fixed_length_right"] = args.fixed_length_right
        match_params["fixed_length_left"] = args.fixed_length_left

        # for claim source
        match_params["use_claim_source"] = args.use_claim_source
        match_params["claim_source_embeddings"] = claim_src_embs_matrix
        # for article source
        match_params["use_article_source"] = args.use_article_source
        match_params["article_source_embeddings"] = article_src_embs_matrix
        # multi-head attention
        match_params["cuda"] = args.cuda
        match_params["num_att_heads_for_words"] = args.num_att_heads_for_words  # first level
        match_params["num_att_heads_for_evds"] = args.num_att_heads_for_evds  # second level

       
        match_params['dropout_gnn'] = args.gnn_dropout
        match_params["dropout_left"] = args.dropout_left
        match_params["dropout_right"] = args.dropout_right
        match_params["hidden_size"] = args.hidden_size

        match_params["gsl_rate"] = args.gsl_rate 

        match_params["embedding_freeze"] = True
        match_params["output_size"] = 2 # if args.dataset == "Snopes" else 3
        match_model = graph_based_semantic_structure.Graph_basedSemantiStructure(match_params)

        file_handler.myprint("Fitting Model")
        fit_model = char_man_fitter_query_repr1.CharManFitterQueryRepr1(net = match_model, loss = args.loss_type, n_iter = args.epochs,
                                                  batch_size = args.batch_size, learning_rate = args.lr,
                                                  early_stopping = args.early_stopping, use_cuda = args.cuda,
                                                  logfolder = outfolder_per_fold, curr_date = curr_date,
                                                  fixed_num_evidences = args.fixed_num_evidences,
                                                  output_handler_fact_checking = file_handler, seed=args.seed,
                                                  output_size=match_params["output_size"])

        try:
            fit_model.fit(train_interactions, verbose = True,  # for printing out evaluation during training
                          # topN = args.topk,
                          val_interactions=valid_interactions,
                          test_interactions=test_interactions)
            dev_results, test_results = fit_model.load_best_model(valid_interactions, test_interactions)
            kfold_dev_results.append(dev_results)
            kfold_test_results.append(test_results)
        except KeyboardInterrupt:
            file_handler.myprint('Exiting from training early')
        t10 = time.time()
        file_handler.myprint('Total time for one fold:  %d (seconds)' % (t10 - t1))

    avg_test_results = evaluator.compute_average_classification_results(kfold_test_results, list_metrics, **additional_data)
    file_handler.myprint("Average results from %s folds" % args.num_folds)
    avg_test_results_json = json.dumps(avg_test_results, sort_keys=True, indent=2)
    file_handler.myprint(avg_test_results_json)
    # save results in json file
    result_json_path = os.path.join(secondary_log_folder, "avg_5fold_result_%s.json" % int(seed))
    with open(result_json_path, 'w') as fin:
        fin.write(avg_test_results_json)
    fin.close()
    ty = time.time()
    file_handler.myprint('Total time:  %d (seconds)' % (ty - tx))
    return avg_test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Description: Running Neural IR Search Models")
    parser.add_argument('--path', default = '../test_code/ben_data_10_no_body', help = 'Input data path', type = str)
    parser.add_argument('--dataset', type = str, default = 'Snopes', help = '[Snopes, Politifact]')
    parser.add_argument('--epochs', default = 100, help = 'Number of epochs to run', type = int)
    parser.add_argument('--batch_size', default = 32, help = 'Batch size', type = int)
    parser.add_argument('--lr', default = 0.001, type = float, help = 'Learning rate')
    parser.add_argument('--early_stopping', default = 10, type = int, help = 'The number of step to stop training')
    parser.add_argument('--debug', default = 1, type = int, help = 'debug or not')
    parser.add_argument('--log', default = "", type = str, help = 'folder for logs and saved models')
    parser.add_argument('--optimizer', nargs = '?', default = 'adam', help = 'Specify an optimizer: adam')

    parser.add_argument('--loss_type', nargs = '?', default = 'cross_entropy', help = 'Specify a loss function')
    parser.add_argument('--word_embedding_size', default = 300, help = 'the dimensions of word embeddings', type = int)
    parser.add_argument('--verbose', type = int, default = 1,  help = 'Show performance per X iterations')
    parser.add_argument('--cuda', type = int, default = 1, help = 'using cuda or not')
    parser.add_argument('--seed', type = int, default = 123456, help = 'random seed')
    parser.add_argument('--decay_step', type = int, default = 100, help = 'how many steps to decay the learning rate')
    parser.add_argument('--decay_weight', type = float, default = 0.0001, help = 'percent of decaying')
    parser.add_argument('--fixed_length_left', type = int, default = 30, help = 'Maximum length of each query')
    parser.add_argument('--fixed_length_right', type = int, default = 100, help = 'Maximum length of each document')

    parser.add_argument('--fixed_num_evidences', type=int, default=30, help='Max number of evidences each claim')
    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds for cross validation')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--use_claim_source', type=int, default=1, help='Using claim source embedding or not')
    parser.add_argument('--use_article_source', type=int, default=1, help = 'Using article source embedding or not')
    parser.add_argument('--num_att_heads_for_words', type = int, default = 1, help = 'Nums att heads for words')
    parser.add_argument('--num_att_heads_for_evds', type = int, default = 1, help = 'Nums att heads for evidences')

    # GNN related
    parser.add_argument('--gnn_window_size', type=int, default=2, help="GNN window size")
    parser.add_argument("--gnn_dropout", type=float, default=0.5, help="Dropout rate for GNN")
    parser.add_argument("--gsl_rate", type=float, default=0.8, help="Dropout rate of GSL")

    parser.add_argument('--dropout_left', type = float, default = 0.2, help = 'Dropout rate for word embs in claim')
    parser.add_argument('--dropout_right', type = float, default = 0.2, help = 'Dropout rate for word embs in articles')
    parser.add_argument('--hidden_size', type=int, default = 300, help = 'Hidden Size of LSTM')

    parser.add_argument('--article_src_emb_size', type = int, default = 128, help = 'Embedding size of article')
    parser.add_argument('--claim_src_emb_size', type = int, default = 128, help = 'Embedding size of article')
    parser.add_argument('--fixed_length_left_src_chars', type = int, default = 20, help = 'Max number of chars for left sources')
    parser.add_argument('--fixed_length_right_src_chars', type = int, default = 20, help = 'Max number of chars for right sources')

   
    args = parser.parse_args()

    avg_test_results = fit_models(args)
