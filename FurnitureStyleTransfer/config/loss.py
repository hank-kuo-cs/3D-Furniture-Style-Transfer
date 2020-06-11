class LossConfig:
    def __init__(self,
                 l_style: float,
                 l_img_compare: float,
                 multiview_loss_func: str):
        self.l_style = l_style
        self.l_img_compare = l_img_compare
        self.multiview_loss_func = multiview_loss_func

        self.check_parameters()

    def check_parameters(self):
        assert isinstance(self.l_style, float)
        assert isinstance(self.l_img_compare, float)

        assert isinstance(self.multiview_loss_func, str)
        assert self.multiview_loss_func == 'L1' or self.multiview_loss_func == 'MSE'

