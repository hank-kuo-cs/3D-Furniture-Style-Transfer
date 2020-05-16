import torch


class CudaConfig:
    def __init__(self,
                 device: str = 'cuda',
                 is_parallel: bool = False,
                 parallel_gpus: list = None):
        self.device = device
        self.is_parallel = is_parallel
        self.parallel_gpus = parallel_gpus

    @property
    def cuda_num(self):
        return torch.cuda.device_count()

    def check_parameters(self):
        assert self.device == 'cuda' or self.device == 'cpu'
        assert isinstance(self.is_parallel, bool)

        if self.device == 'cuda':
            assert torch.cuda.is_avalibale()



