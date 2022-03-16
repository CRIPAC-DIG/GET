from tensorboardX import SummaryWriter

class TensorboardWrapperClass(object):
    # my_tensorboard_writer = None

    def __init__(self):
        pass

    # @classmethod
    def init_log_files(self, log_file):
        if log_file != None:
            self.my_tensorboard_writer = SummaryWriter(log_file)

    # @classmethod
    def mywriter(self):
        assert self.my_tensorboard_writer != None, "The LogFile is not initialized yet!"
        return self.my_tensorboard_writer