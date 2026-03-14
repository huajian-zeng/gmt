<p align="center">
  <h1 align="center"><strong>GMT: Goal-Conditioned Multimodal Transformer for 6-DOF Object Trajectory Synthesis in 3D Scenes</strong></h1>
  <p align="center">
    <a href="https://huajian-zeng.github.io/">Huajian Zeng</a><sup>1,4</sup>, <a href="https://cvg.cit.tum.de/members/saroha">Abhishek Saroha</a><sup>1,2</sup>, <a href="https://vision.in.tum.de/members/cremers">Daniel Cremers</a><sup>1,2</sup>, <a href="https://xiwang1212.github.io/homepage/">Xi Wang</a><sup>1,2,3</sup>
    <br>
    <sup>1</sup>Technical University of Munich (TUM), <sup>2</sup>Munich Center for Machine Learning (MCML), <sup>3</sup>ETH Zürich, <sup>4</sup>Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)
    <br>
  </p>

  <p align="center"><strong>3DV 2026</strong></p>
</p>

<div id="top" align="center">

[![PDF](https://img.shields.io/badge/PDF-%F0%9F%93%84-green)](https://huajian-zeng.github.io/projects/gmt/3dv2026_arxiv.pdf)
[![Homepage](https://img.shields.io/badge/Homepage-%F0%9F%8C%90-blue)](https://huajian-zeng.github.io/projects/gmt/)
[![Video](https://img.shields.io/badge/Video-%E2%96%B6-red?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=uYZzJsfn_T4&feature=youtu.be)

</div>


## 🔥 Highlight <a name="highlight"></a>

**GMT** is a goal-conditioned multimodal transformer that synthesizes physically plausible 6-DOF object trajectories in 3D scenes from egocentric observations.

By leveraging multimodal conditioning — combining 3D scene geometry, egocentric video features, and goal specifications — GMT generates temporally coherent and spatially accurate object motion sequences.


## Installation

1. Create a new conda environment with Python 3.9:
    ```bash
    conda create -n gmt python=3.9 -y
    ```
2. Activate the conda environment:
    ```bash
    conda activate gmt
    ```
3. Install PyTorch and torchvision. Please refer to the [official PyTorch website](https://pytorch.org/get-started/locally/) for the appropriate command based on your CUDA version.
    ```bash
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    ```
4. Install other dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. Install PointNet++:
    ```bash
    git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
    cd Pointnet2_PyTorch
    pip install -e pointnet2_ops_lib --no-build-isolation
    pip install -r requirements.txt
    pip install -e .
    # [IMPORTANT] you need to change l196-198 of file `[PATH-TO-VENV]/lib64/python3.9/site-packages/pointnet2_ops/pointnet2_modules.py` to `interpolated_feats = known_feats.repeat(1, 1, unknown.shape[1])`)
    ```

## Dataset Preparation

- Option 1: Download preprocessed cache files by running the following command:
    ```bash
    bash scripts/download_adt_cache.sh
    ```
- Option 2:
Please follow the instructions in the [ADT official website](https://www.projectaria.com/datasets/adt/) to download the ADT dataset using [projectaria_tools](https://github.com/facebookresearch/projectaria_tools). The downloaded data follows the `projectaria_tools_adt_data` format. Here is an example sequence structure:
    ```bash
    projectaria_tools_adt_data/
        Apartment_release_work_seq136_M1292/ # example sequence
            2d_bounding_box_with_skeleton_annotations/
            3d_bounding_box_annotations/
            depth_images/
            instances/
            segmentations/
            synthetic/
            video/
            ...
    ```

## Download Pretrained Checkpoint
```bash
bash scripts/download_pretrained.sh
```

## Quick Start

Here is an example command to run inference on a single sequence from the ADT dataset using a pretrained model:

```bash
bash demo.sh
```

## Training

### Training on ADT dataset:

```bash
python -m scripts.train --adt_dataroot <path_to_raw_data>
```
You may also define:
- `--global_cache_dir`: path to the trajectory cache directory. If not provided, it will be created in the `save_path` directory.
- `--save_path`: path to save training results.
- `--exp_name`: experiment name for checkpoint folder.
- `--train_split_file` and `--val_split_file`: text files containing the list of training and validation sequences, respectively.
- `--wandb_mode`: set to `disabled` to disable wandb logging.

## Evaluation

```bash
python -m scripts.eval \
    --adt_dataroot <path_to_raw_data> \
    --model_path <path_to_trained_model_checkpoint>
```

You may also define:
- `--global_cache_dir`: path to the trajectory cache directory.
- `--max_eval_samples`: maximum number of evaluation samples. Default is -1 (use all samples).

## Citation
If you find this repository useful for your research, please consider citing:
```bibtex
@inproceedings{zeng2026gmt,
  title     = {{GMT}: Goal-Conditioned Multimodal Transformer for 6-DOF Object Trajectory Synthesis in 3D Scenes},
  author    = {Zeng, Huajian and Saroha, Abhishek and Cremers, Daniel and Wang, Xi},
  booktitle = {International Conference on 3D Vision (3DV)},
  year      = {2026},
}
```

## Acknowledgements
We adapted code from several excellent repositories, including:

- [PerciverIO](https://github.com/krasserm/perceiver-io)
- [GIMO](https://github.com/y-zheng18/GIMO)

We sincerely thank the authors for open-sourcing their work.
