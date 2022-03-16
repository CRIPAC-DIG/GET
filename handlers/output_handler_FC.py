import sys

class FileHandlerFC(object):
    # mylogfile = None
    # mylogfile_details = None
    # error_analysis_log_validation = None
    # error_analysis_log_testing = None
    # error_analysis_log_test2 = None
    # error_analysis_log_test3 = None

    def __init__(self):
        pass

    # @classmethod
    def init_log_files(self, log_file):
        if log_file != None:
            self.mylogfile = open(log_file, "w")
            self.mylogfile_details = open(log_file + "_best_details.json", "w")
            self.error_analysis_log_validation = open(log_file + "_error_analysis_validation.json", "w")
            self.error_analysis_log_testing = open(log_file + "_error_analysis_testing.json", "w")
            self.error_analysis_log_test2 = open(log_file + "_error_analysis_test2.json", "w")
            self.error_analysis_log_test3 = open(log_file + "_error_analysis_test3.json", "w")

    # @classmethod
    def myprint(self, message):
        assert self.mylogfile != None, "The LogFile is not initialized yet!"
        print(message)
        sys.stdout.flush()
        if self.mylogfile != None:
            print(message, file = self.mylogfile)
            self.mylogfile.flush()

    # @classmethod
    def myprint_details(self, message):
        assert self.mylogfile_details != None, "The Detailed JSON log file is not initialized yet!"
        # print(message)
        if self.mylogfile_details != None:
            print(message, file = self.mylogfile_details)
            self.mylogfile_details.flush()

    # @classmethod
    def save_error_analysis_validation(self, message: str):
        assert self.error_analysis_log_validation != None, "The Detailed JSON log file is not initialized yet!"
        # print(message)
        if self.error_analysis_log_validation != None:
            print(message, file = self.error_analysis_log_validation)
            self.error_analysis_log_validation.flush()

    # @classmethod
    def save_error_analysis_testing(self, message: str):
        assert self.error_analysis_log_testing != None, "The Detailed JSON log file is not initialized yet!"
        # print(message)
        if self.error_analysis_log_testing != None:
            print(message, file = self.error_analysis_log_testing)
            self.error_analysis_log_testing.flush()

    # @classmethod
    def save_error_analysis_test2(self, message: str):
        assert self.error_analysis_log_test2 != None, "The Detailed JSON log file is not initialized yet!"
        # print(message)
        if self.error_analysis_log_test2 != None:
            print(message, file=self.error_analysis_log_test2)
            self.error_analysis_log_test2.flush()

    # @classmethod
    def save_error_analysis_test3(self, message: str):
        assert self.error_analysis_log_test3 != None, "The Detailed JSON log file is not initialized yet!"
        # print(message)
        if self.error_analysis_log_test3 != None:
            print(message, file=self.error_analysis_log_test3)
            self.error_analysis_log_test3.flush()

    def close(self):
        self.mylogfile.close()
        self.mylogfile_details.close()
        self.error_analysis_log_validation.close()
        self.error_analysis_log_testing.close()
        self.error_analysis_log_test2.close()
        self.error_analysis_log_test3.close()