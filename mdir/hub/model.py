import torch
import sys
from pathlib import Path


BASE_URL = "http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/"
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from mdir.learning.checkpoints import Checkpoints
from mdir.learning.network import initialize_network, SingleNetwork
from mdir.components.data.transform import initialize_transforms
from mdir.tools.utils import load_yaml_scenario


def _create(path, substitutions):
    params = load_yaml_scenario([path])
    for target, value in substitutions.items():
        p = params
        for k in target.split(".")[:-1]:
            p = p[k]
        p[target.split(".")[-1]] = value
    state = Checkpoints.load_network(params["path"])
    # Fix cirnet unnecessary download
    if state["net"]["network_params"]["model"]["architecture"] == "cirnet":
        state["net"]["network_params"]["model"]["pretrained"] = False
    device = torch.device("cpu")
    network = initialize_network(None, device, state, params["runtime"]).eval()

    data_params = state["net"]["network_params"]["runtime"]["data"]
    # Fix transforms to augmentations arg
    if "augmentations" not in data_params:
        data_params["augmentations"] = data_params.pop("transforms")
    network.__setattr__("transform", initialize_transforms(**data_params))

    return network


#
# ICCV23 embedding networks
#


def gem_vgg16_cyclegan():
    return _create(
        str(FILE.parent / "embedding.yml"), {
            "path": f"{BASE_URL}cyclegan_embed_vgg16.pth",
            "runtime.wrappers.eval.0_cirwhiten.whitening": f"{BASE_URL}cyclegan_embed_vgg16_lw.pkl"
        }
    )


def gem_vgg16_hedngan():
    return _create(
        str(FILE.parent / "embedding.yml"), {
            "path": f"{BASE_URL}hedngan_embed_vgg16.pth",
            "runtime.wrappers.eval.0_cirwhiten.whitening": f"{BASE_URL}hedngan_embed_vgg16_lw.pkl"
        }
    )


def gem_resnet101_cyclegan():
    return _create(
        str(FILE.parent / "embedding.yml"), {
            "path": f"{BASE_URL}cyclegan_embed_resnet101.pth",
            "runtime.wrappers.eval.0_cirwhiten.whitening": f"{BASE_URL}cyclegan_embed_resnet101_lw.pkl"
        }
    )


def gem_resnet101_hedngan():
    return _create(
        str(FILE.parent / "embedding.yml"), {
            "path": f"{BASE_URL}hedngan_embed_resnet101.pth",
            "runtime.wrappers.eval.0_cirwhiten.whitening": f"{BASE_URL}hedngan_embed_resnet101_lw.pkl"
        }
    )


#
# ICCV23 generator networks
#


def generator_cyclegan():
    return _create(str(FILE.parent / "generator.yml"), {"path": f"{BASE_URL}cyclegan_generator_X.pth"})


def generator_hedngan():
    return _create(str(FILE.parent / "generator.yml"), {"path": f"{BASE_URL}hedngan_generator_X.pth"})