from pytorch3d.renderer import DirectionalLights
from ....config import config


def load_lights():
    return DirectionalLights(device=config.cuda.device,
                             direction=((-40.0, 200.0, 100.0),),
                             ambient_color=((0.7, 0.7, 0.7),),
                             diffuse_color=((0.8, 0.8, 0.8),),
                             specular_color=((0.0, 0.0, 0.0),))
