import torch


class CudaConfig:
    def __init__(self,
                 device: str = 'cuda',
                 cuda_num: int = 0,
                 is_parallel: bool = False,
                 parallel_gpus: list = None):
        self.device = device
        self.cuda_num = cuda_num
        self.is_parallel = is_parallel
        self.parallel_gpus = parallel_gpus

    @property
    def cuda_count(self):
        return torch.cuda.device_count()

    def check_parameters(self):
        assert self.device == 'cuda' or self.device == 'cpu'
        assert isinstance(self.is_parallel, bool)
        assert isinstance(self.cuda_num, int) and self.cuda_num >= 0

        if self.device == 'cuda':
            assert torch.cuda.is_avalibale()



