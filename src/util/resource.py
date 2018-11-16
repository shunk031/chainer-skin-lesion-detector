from logzero import logger


class Resource(object):

    def __init__(self, args, train=False):
        self.args = args
        self.train = train
        self.logger = logger

        self.common_info()

    def log_debug(self, msg):
        return self.logger.debug(msg)

    def log_info(self, msg):
        return self.logger.info(msg)

    def common_info(self):

        self.log_info(f'Model: {self.args.model}')
        self.log_info(f'Batchsize: {self.args.batchsize}')
        self.log_info(f'GPU ID: {self.args.gpu}')
        self.log_info(f'Output directory: {self.args.out}')
        self.log_info(f'Resume: {self.args.resume}')
