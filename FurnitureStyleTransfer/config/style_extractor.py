class StyleExtractorConfig:
    def __init__(self,
                 network_model: str,
                 feature_margin: float,
                 feature_dim: int,
                 batch_size: int,
                 epoch_num: int,
                 lr: float,
                 momentum: float,
                 weight_decay: float):

        self.network_model = network_model
        self.feature_margin = feature_margin
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
