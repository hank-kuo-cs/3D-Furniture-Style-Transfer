import torch
import numpy as np
from pytorch3d.renderer import look_at_view_transform, OpenGLPerspectiveCameras
from ....config import config


def load_perspective_cameras():
    device = torch.device(config.cuda.device)

    return OpenGLPerspectiveCameras(device=device,
                                    degrees=True,
                                    fov=30,
                                    znear=0.00001,
                                    zfar=10000)


def load_camera_positions(dist, elev, azim, at, up):
    device = torch.device(config.cuda.device)

    pi = np.pi
    assert -pi * 0.5 <= elev <= pi * 0.5
    assert 0 <= azim <= pi * 2

    R, T = look_at_view_transform(dist=dist,
                                  elev=elev,
                                  azim=azim,
                                  degrees=False,
                                  at=((at[0], at[1], at[2]),),
                                  up=((up[0], up[1], up[2]),),
                                  device=device)

    return R, T
