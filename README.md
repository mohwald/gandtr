# GAN domain translation for recognition

**[arXiv](https://arxiv.org/abs/2309.16351) | [Paper with supplementary](https://openaccess.thecvf.com/content/ICCV2023/html/Mohwald_Dark_Side_Augmentation_Generating_Diverse_Night_Examples_for_Metric_Learning_ICCV_2023_paper.html) | [Video (5m)](https://youtu.be/zlT-GJOcgYw)**

----

Codebase for the publication:

**Dark Side Augmentation**: Generating Diverse Night Examples for Metric Learning [[arXiv](https://arxiv.org/abs/2309.16351)].
Albert Mohwald, [Tomas Jenicek][jenicek] and [Ondřej Chum][chum].
In International Conference on Computer Vision (ICCV), 2023.

This repository builds on top of image retrieval implemented in [mdir][mdir] and [cirtorch][cirtorch] and adapts [CycleGAN][cyclegan] and [CUT][cut] for image-to-image translation.

![train_and_finetune](https://github.com/mohwald/gandtr/assets/29608815/bc284d8a-5da6-4e24-921a-28717e9015e1)

----

## Pretrained models

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Weights</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>CycleGAN (day-to-night)</td>
      <td><a href="http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/cyclegan_generator_X.pth">Download</a></td>
    </tr>
    <tr>
      <td>HED<sup>N</sup>GAN (day-to-night)</td>
      <td><a href="http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/hedngan_generator_X.pth">Download</a></td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Avg</th>
      <th>Tokyo</th>
      <th>ROxf</th>
      <th>RPar</th>
      <th>Weights</th>
      <th>Whitening</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GeM VGG16 CycleGAN</td>
      <td>74.0</td>
      <td>90.2</td>
      <td>60.7</td>
      <td>71.0</td>
      <td><a href="http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/cyclegan_embed_vgg16.pth">Download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/cyclegan_embed_vgg16_lw.pkl">Download</a></td>
    </tr>
    <tr>
      <td>GeM VGG16 HED<sup>N</sup>GAN</td>
      <td>73.5</td>
      <td>88.8</td>
      <td>61.1</td>
      <td>70.7</td>
      <td><a href="http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/hedngan_embed_vgg16.pth">Download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/hedngan_embed_vgg16_lw.pkl">Download</a></td>
    </tr>
    <tr>
      <td>GeM ResNet-101 CycleGAN</td>
      <td>78.4</td>
      <td>92.0</td>
      <td>66.8</td>
      <td>76.4</td>
      <td><a href="http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/cyclegan_embed_resnet101.pth">Download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/cyclegan_embed_resnet101_lw.pkl">Download</a></td>
    </tr>
    <tr>
      <td>GeM ResNet-101 HED<sup>N</sup>GAN</td>
      <td>78.4</td>
      <td>91.7</td>
      <td>66.6</td>
      <td>76.8</td>
      <td><a href="http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/hedngan_embed_resnet101.pth">Download</a></td>
      <td><a href="http://ptak.felk.cvut.cz/personal/jenicto2/download/iccv23_gan/hedngan_embed_resnet101_lw.pkl">Download</a></td>
    </tr>
  </tbody>
</table>

All models are pretrained on [Retrieval-SfM 120k][sfm].

### Torch Hub

To use any pretrained model, please follow [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

```python
import torch

# Day-to-night generators
cyclegan = torch.hub.load('mohwald/gandtr', 'cyclegan', pretrained=True)
hedngan = torch.hub.load('mohwald/gandtr', 'hedngan', pretrained=True)

# Image descriptors
gem_vgg16_cyclegan = torch.hub.load('mohwald/gandtr', 'gem_vgg16_cyclegan', pretrained=True)
gem_vgg16_hedngan = torch.hub.load('mohwald/gandtr', 'gem_vgg16_hedngan', pretrained=True)
gem_resnet101_cyclegan = torch.hub.load('mohwald/gandtr', 'gem_resnet101_cyclegan', pretrained=True)
gem_resnet101_hedngan = torch.hub.load('mohwald/gandtr', 'gem_resnet101_hedngan', pretrained=True)
```

Models initialized this way are pretrained and loaded on GPU by default. If do not want to load pretrained weights, pass `pretrained=False`; to load the model on e.g. CPU, pass `device="cpu"`. 

> [!IMPORTANT]
> The expected input of all descriptor models listed above is a **batch of normalized images after CLAHE transform**. To recommended way how to obtain the image preprocessing transforms (suitable for dataset loader) is demonstrated in the snippet below:

```
>>> import torch
>>> model = torch.hub.load('mohwald/gandtr', 'gem_vgg16_hedngan')
>>> model.transform
Compose(
    Pil2Numpy()
    ApplyClahe(clip_limit=1.0, grid_size=8, colorspace=lab)
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], strict_shape=True)
)
```

----


## Installation

1. Install ffmpeq and graphviz, if you do not have it;  ffmpeg is required for OpenCV, graphviz allows to draw network architecture. 
```
sudo apt-get install ffmpeg libsm6 libxext6 graphviz   # Ubuntu and Debian-based system
sudo dnf install ffmpeg libSM graphviz                 # RHEL, Fedora, and CentOS-based system 
```
2. Clone this repository: `git@github.com:mohwald/gandtr.git && cd gandtr`
3. Install dependencies: `pip install -r requirements.txt`
4. (Optional) set environment variables:
    - `${CIRTORCH_ROOT}`, where you want to store image data and model weights. All necesarry data for the evaluation and training are automatically downloaded there.
    - `${CUDA_VISIBLE_DEVICES}`, set to a single gpu, e.g. `export CUDA_VISIBLE_DEVICES=0`
5. Go to `mdir/examples`


## General scenario format and execution

Inside `mdir/examples`, each experiment can be executed by script `perform_scenario.py`, that runs yaml scenarios based on this structure:
```yaml
TARGET:
  1_step:  # first step parameters dictionary
      ...
  2_step:  # second step parameters dictionary
      ...
  ...
```

Nested dictionary keys can be used in parameters and variables (nested keys are separated by a dot).
Bash-style variables are supported within a TARGET, e.g. `${1_step.section.key}`.
A special variable `${SCENARIO_NAME}` denotes the name of the executed scenario (last scenario name, if scenarios are overlayed).

A scenario is executed with `perform_scenario.py` as:
```bash
python3 perform_scenario.py TARGET SCENARIO_NAME_1.yml [SCENARIO_NAME_2.yml]...
```

Scenarios can overlay, so that all variables of `SCENARIO_NAME_1` are replaced by variables from `SCENARIO_NAME_2`.


## Evaluation

All scenarios for evaluation are located inside `iccv23/eval`.

To evaluate a model from ICCV23 paper, e.g. HED-N-GAN method with GeM VGG16 backbone, run:

```bash
python3 perform_scenario.py eval iccv23/eval/hedngan.yml
```

> [!WARNING]
> Oxford and Paris buildings dataset images are no longer available at the original sources and thus cannot be downloaded automatically. One option is to download images from [Kaggle](https://www.kaggle.com/datasets/skylord/oxbuildings) (requires registration). Images should be placed inside `${CIRTORCH_ROOT}/data/test/{oxford5k, paris6k}/jpg`, without any nested directories.

To change the GAN generator used in the augmentation, use different scenario with the corresponding generator name.
To change the embedding backbone, change `eval` to `eval_r101` to evaluate on GeM ResNet-101.
With these options, you should get the following results:

VGG-16 Backbone (`eval`)

| Model       | Tokyo | ROxf | RPar |
|-------------|-------|------|------|
| hedngan     | 88.8  | 61.1 | 70.7 |
| cyclegan    | 90.2  | 60.7 | 71.0 |

ResNet-101 Backbone (`eval_r101`)

| Model       | Tokyo | ROxf | RPar |
|-------------|-------|------|------|
| hedngan     | 91.7  | 66.6 | 76.8 |
| cyclegan    | 92.0  | 66.8 | 76.4 |


## Training

All scenarios for training from scratch are located inside `iccv23/train`.

### GAN generator training

To train a GAN generator from scratch, e.g. HED-N-GAN, run:

```bash
python3 perform_scenario.py train iccv23/train/hedngan.yml
```

To change the GAN model, replace the yaml scenario with the scenario corresponding to the model name, e.g. `hedngan.yml` with `cyclegan.yml`, etc.

(Optional) After the generator training is finished, arbitrary images can be outputted by the trained generator given a list of image paths from standard input and executing the `output` target:

```bash
python3 perform_scenario.py output iccv23/train/hedngan.yml
```

### Metric learning

To finetune an embedding network for image retrieval, which uses augmentation with HED-N-GAN generator, run:

```bash
python3 perform_scenario.py finetune iccv23/train/hedngan.yml
```

This command will both finetune the embedding model and consequently evaluate it.

To change the backbone used for the finetuning, replace `finetune` with `finetune_r101` for GeM ResNet-101.

<!-- References -->

[jenicek]: http://cmp.felk.cvut.cz/~jenicto2
[chum]: http://cmp.felk.cvut.cz/~chum
[mdir]: https://github.com/jenicek/mdir/
[cirtorch]: https://github.com/filipradenovic/cnnimageretrieval-pytorch/
[cyclegan]: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/
[cut]: https://github.com/taesungp/contrastive-unpaired-translation/
[sfm]: https://cmp.felk.cvut.cz/cnnimageretrieval/
