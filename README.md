# GAN domain translation for recognition

Codebase for the publication:

**Dark Side Augmentation**: Generating Diverse Night Examples for Metric Learning  
Albert Mohwald, [Tomas Jenicek][jenicek] and [Ondřej Chum][chum]  
In International Conference on Computer Vision (ICCV), 2023

This repository is a fork of [mdir](https://github.com/jenicek/mdir/), builds on top of [cirtorch](https://github.com/filipradenovic/cnnimageretrieval-pytorch/) for image retrieval, and adapts [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [CUT](https://github.com/taesungp/contrastive-unpaired-translation) for image-to-image translation.

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

> **Warning**<br>
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