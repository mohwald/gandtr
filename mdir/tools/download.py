import os
import urllib.request
import warnings

import torch.utils.model_zoo as model_zoo
from pathlib import Path

from ..external.cirtorch.utils.download import download_train, download_test
from ..external.cirtorch.utils.general import get_data_root


VAL_IMS = [
    "8a/b3/ab/3fb7b4f3c9560312f1e36f00a7abb38a",
    "39/f6/93/015bb8aa57c3fbebc41daca6a093f639",
    "37/67/5f/0030a526c20a135a33e93d0d495f6737",
    "81/fd/18/d543f7828509ebc931c80134b818fd81",
]

WEIGHTS_ROOT = os.path.join(get_data_root(), "..", "weights")


def rsfm120k(data_dir):
    # Download Retrieval-SfM-120k
    download_train(data_dir)
    # Download GAN training dataset
    download_files(["train_day.txt", "train_night.txt"],
                   os.path.join(data_dir, "train", "retrieval-SfM-120k", "dataset"),
                   "http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/")
    # Prepare images for visual validation
    img_dir = _create_dir(os.path.join(data_dir, "val", "day_night"))
    for i, img in enumerate(VAL_IMS, 1):
        src = os.path.join(data_dir, "train", "retrieval-SfM-120k", "ims", img)
        dest = os.path.join(img_dir, str(i)) + ".jpg"
        _link(src, dest)


def roxf5k_rpar6k_247tokyo1k(data_dir):
    # Download Revisited Oxford and Paris, 24/7 Tokyo datasets
    download_test(data_dir)
    # Hint if Oxford and Paris do not have images available
    if len(list((Path(data_dir) / "data" / "test" / "oxford5k" / "jpg").glob(".jpg"))) == 0 \
        or len(list((Path(data_dir) / "data" / "test" / "paris6k" / "jpg").glob(".jpg"))) == 0:
        warnings.warn("Images for oxford5k or paris6k are missing. Please, register at Kaggle and download images\n"
                      "at https://www.kaggle.com/datasets/skylord/oxbuildings \n"
                      "Then, extract the images into the their corresponding datasets with the following commands:\n"
                      "\n"
                      "unzip archive.zip\n"
                      "rm archive.zip\n"
                      "mkdir --parents ${CIRTORCH_ROOT}/data/test/oxford5k/jpg\n"
                      "tar -xzvf oxbuild_images.tgz -C ${CIRTORCH_ROOT}/data/test/oxford5k/jpg/\n"
                      "mkdir --parents ${CIRTORCH_ROOT}/data/test/paris6k/jpg\n"
                      "tar -xzvf paris_1.tgz -C ${CIRTORCH_ROOT}/data/test/paris6k/jpg/\n"
                      "tar -xzvf paris_2.tgz -C ${CIRTORCH_ROOT}/data/test/paris6k/jpg/\n"
                      "mv ${CIRTORCH_ROOT}/data/test/paris6k/jpg/paris/*/*.jpg ${CIRTORCH_ROOT}/data/test/paris6k/jpg/\n"
                      "rm -rf ${CIRTORCH_ROOT}/data/test/paris6k/jpg/paris\n"
                      "rm oxbuild_images.tgz\n"
                      "rm paris_1.tgz\n"
                      "rm paris_2.tgz\n"
                      "\n"
                      "Perform this scenario again.")


def download_files(names, root_path, base_url, logfunc=None):
    """Download file names from given url to given directory path. If logfunc given, use it to log
        status."""
    root_path = Path(root_path)
    for name in names:
        path = root_path / name
        if path.exists():
            continue
        if logfunc:
            logfunc(f"Downloading file '{name}'")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(base_url + name, path)


def download_and_load_pretrained(url):
    return model_zoo.load_url(url, model_dir=WEIGHTS_ROOT, progress=True, map_location=lambda storage, loc: storage)


def _create_dir(d):
    if not os.path.isdir(d):
        Path(d).mkdir(parents=True, exist_ok=True)
    return d


def _link(src, dest):
    if not os.path.islink(dest):
        _create_dir(os.path.dirname(dest))
        Path(dest).symlink_to(src)
    return dest
