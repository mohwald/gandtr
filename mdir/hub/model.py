import torch
import sys
from pathlib import Path

BASE_URL = "http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/"
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1].parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mdir.learning.checkpoints import Checkpoints
from mdir.learning.network import initialize_network
from mdir.components.data.transform import initialize_transforms
from mdir.tools.utils import load_yaml_scenario


def _create(path, substitutions, pretrained):
    params = load_yaml_scenario([str(path)])
    params = params["pretrained"] if pretrained else params["initialized"]
    for target, value in substitutions.items():
        p = params
        for k in target.split(".")[:-1]:
            p = p[k]
        p[target.split(".")[-1]] = value

    device = torch.device("cpu")
    if pretrained:
        state = Checkpoints.load_network(params["path"])
        # Fix cirnet unnecessary download
        if state["net"]["network_params"]["model"]["architecture"] == "cirnet":
            state["net"]["network_params"]["model"]["pretrained"] = False
        network = initialize_network(None, device, state, params["runtime"]).eval()
    else:
        network = initialize_network(params, device).eval()

    data_params = network.network_params.runtime["data"]
    # Fix transforms to augmentations arg
    if "augmentations" not in data_params:
        data_params["augmentations"] = data_params.pop("transforms")
    network.__setattr__("transform", initialize_transforms(**data_params))

    return network


#
# ICCV23 embedding networks
#


def gem_vgg16_cyclegan(pretrained=True):
    """
    GeM global descriptor model with VGG16 backbone (optionally) pretrained on Retrieval-SfM 120k dataset
    with CycleGAN query augmentation and with CLAHE.
    """
    if pretrained:
        return _create(
            FILE.parent / "embedding.yml", {
                "path": f"{BASE_URL}cyclegan_embed_vgg16.pth",
                "runtime.wrappers.eval.0_cirwhiten.whitening": f"{BASE_URL}cyclegan_embed_vgg16_lw.pkl"
            },
            pretrained
        )
    return _create(FILE.parent / "embedding.yml", {"model.cir_architecture": "vgg16"}, pretrained)


def gem_vgg16_hedngan(pretrained=True):
    """
    GeM global descriptor model with VGG16 backbone (optionally) pretrained on Retrieval-SfM 120k dataset
    with HED-N-GAN query augmentation and with CLAHE.
    """
    if pretrained:
        return _create(
            FILE.parent / "embedding.yml", {
                "path": f"{BASE_URL}hedngan_embed_vgg16.pth",
                "runtime.wrappers.eval.0_cirwhiten.whitening": f"{BASE_URL}hedngan_embed_vgg16_lw.pkl"
            },
            pretrained
        )
    return _create(FILE.parent / "embedding.yml", {"model.cir_architecture": "vgg16"}, pretrained)


def gem_resnet101_cyclegan(pretrained=True):
    """
    GeM global descriptor model with ResNet-101 backbone (optionally) pretrained on Retrieval-SfM 120k dataset
    with CycleGAN query augmentation and with CLAHE.
    """
    if pretrained:
        return _create(
            FILE.parent / "embedding.yml", {
                "path": f"{BASE_URL}cyclegan_embed_resnet101.pth",
                "runtime.wrappers.eval.0_cirwhiten.whitening": f"{BASE_URL}cyclegan_embed_resnet101_lw.pkl"
            },
            pretrained
        )
    return _create(FILE.parent / "embedding.yml", {"model.cir_architecture": "resnet101"}, pretrained)


def gem_resnet101_hedngan(pretrained=True):
    """
    GeM global descriptor model with ResNet-101 backbone (optionally) pretrained on Retrieval-SfM 120k dataset
    with HED-N-GAN query augmentation and with CLAHE.
    """
    if pretrained:
        return _create(
            FILE.parent / "embedding.yml", {
                "path": f"{BASE_URL}hedngan_embed_resnet101.pth",
                "runtime.wrappers.eval.0_cirwhiten.whitening": f"{BASE_URL}hedngan_embed_resnet101_lw.pkl"
            },
            pretrained
        )
    return _create(FILE.parent / "embedding.yml", {"model.cir_architecture": "resnet101"}, pretrained)


#
# ICCV23 generator networks
#


def cyclegan(pretrained=True):
    """
    ResNet CycleGAN generator (optionally) pretrained on Retrieval-SfM 120k dataset for day-to-night image translation.
    """
    if pretrained:
        return _create(FILE.parent / "generator.yml", {"path": f"{BASE_URL}cyclegan_generator_X.pth"}, pretrained)
    return _create(FILE.parent / "generator.yml", {}, pretrained)


def hedngan(pretrained=True):
    """
    ResNet HED-N-GAN generator (optionally) pretrained on Retrieval-SfM 120k dataset for day-to-night image translation.
    """
    if pretrained:
        return _create(FILE.parent / "generator.yml", {"path": f"{BASE_URL}hedngan_generator_X.pth"}, pretrained)
    return _create(
        FILE.parent / "generator.yml",
        {"model.norm_layer": "batch", "initialize.weights": "kaiming_p2p"},
        pretrained
    )
