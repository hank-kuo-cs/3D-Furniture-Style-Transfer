class StyleExtractorConfig:
    def __init__(self,
                 network_model: str,
                 feature_margin: float,
                 feature_dim: int,
                 batch_size: int,
                 epoch_num: int,
                 optimizer: str,
                 lr: float,
                 momentum: float,
                 weight_decay: float):

        self.network_model = network_model
        self.feature_margin = feature_margin
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.check_parameters()

    def check_parameters(self):
        assert isinstance(self.network_model, str)
        assert isinstance(self.feature_margin, float)
        assert isinstance(self.feature_dim, int)
        assert isinstance(self.batch_size, int)
        assert isinstance(self.epoch_num, int)
        assert isinstance(self.optimizer, str)
        assert isinstance(self.lr, float)
        assert isinstance(self.momentum, float)
        assert isinstance(self.weight_decay, float)

        assert self.feature_margin > 0
        assert self.feature_dim > 0
        assert self.batch_size > 0
        assert self.epoch_num > 0
        assert self.optimizer == 'SGD' or self.optimizer == 'Adam'
        assert self.lr >= 0
        assert self.momentum >= 0
        assert self.weight_decay >= 0
