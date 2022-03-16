import torch
import numpy as np
import torch_utils
from Models import base_model
import losses as my_losses
import torch_utils as my_utils
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import os
from handlers import output_handler, mz_sampler
from Evaluation import mzEvaluator as my_evaluator
import datetime
import json
import matchzoo
import interactions
from handlers.tensorboard_writer import TensorboardWrapper
from setting_keywords import KeyWordSettings
from matchzoo.metrics import average_precision, discounted_cumulative_gain, \
    mean_average_precision, mean_reciprocal_rank, normalized_discounted_cumulative_gain, precision


class DenseBaselineFitter:

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

        """I put this fit function here for temporarily """
        assert loss in KeyWordSettings.LOSS_FUNCTIONS
        self._loss = loss
        # self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._testing_epochs = testing_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._reg_l2 = reg_l2
        # self._decay_step = decay_step
        # self._decay_weight = decay_weight

        self._optimizer_func = optimizer_func

        self._use_cuda = use_cuda
        self._num_negative_samples = num_negative_samples
        self._early_stopping_patience = early_stopping # for early stopping

        self._n_users, self._n_items = None, None
        self._net = net
        self._optimizer = None
        # self._lr_decay = None
        self._loss_func = None
        assert logfolder != ""
        self.logfolder = logfolder
        if not os.path.exists(logfolder):
            os.mkdir(logfolder)

        self.saved_model = os.path.join(logfolder, "saved_model_%s" % seed)
        TensorboardWrapper.init_log_files(os.path.join(logfolder, "tensorboard_%s" % seed))
        # for evaluation during training
        self._sampler = mz_sampler.Sampler()
        self._candidate = dict()

    def __repr__(self):
        """ Return a string of the model when you want to print"""
        # todo
        return "Vanilla matching Model"

    def _initialized(self):
        return self._net is not None



