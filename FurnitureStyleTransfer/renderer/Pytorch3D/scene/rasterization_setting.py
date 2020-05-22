import numpy as np
from pytorch3d.renderer import RasterizationSettings


def load_rasterization_setting():
    return RasterizationSettings(image_size=256,
                                 blur_radius=np.log(1. / 1e-4 - 1.) * 1e-4,
                                 faces_per_pixel=100)
