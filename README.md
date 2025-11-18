# ChargeDIFF

[[`Paper`](Tobeupdated)]
[[`Project Page`](https://parkjunkil.github.io/ChargeDIFF/)]

Code release for the paper [`Integrating electronic structure into generative modeling of inorganic materials`]()

<p align="center">
  <img src="docs/objects/Fig1.jpg" alt="Fig1" width="700">
</p>

ChargeDIFF is the first generative model for inorganic materials that explicitly considers electronic structure of materials. Moving beyond conventional structure-only representations (atom types, atomic coordinates, lattice), ChargeDIFF incorporates charge density as a additaional modality during the generation process. ChargeDIFF exhibited superior generation performance for bot h unconditional and conditional generation tasks, and ablation study attribute these gains to the model's deeper understanding on materials' electronic behavior. Please visit [Project Page](https://parkjunkil.github.io/ChargeDIFF/) for more details.

<p align="center">
  <img src="docs/objects/uncond.gif"  width="250">
</p>

# Installation
We recommend to build a [`conda`](https://www.anaconda.com/products/distribution) environment as follow. You might need a different version of `cudatoolkit` depending on your GPU driver.
```
conda create -n chargediff python=3.10 -y && conda activate chargediff
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torch-geometric
pip install h5py joblib termcolor scipy einops tqdm matplotlib opencv-python PyMCubes imageio trimesh omegaconf tensorboard notebook Pillow==9.5.0 py3Dmol ipywidgets transformers pormake p_tqdm pyxtal matminer
conda install -y cudatoolkit=11.7 -c conda-forge
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install numpy=1.23.5
```

# Test ChargeDIFF

## Download the pretrained weight

First create a folder `./saved_ckpt` to save the pre-trained weights. Then download the pre-trained weights from the provided links and put them in the `./saved_ckpt` folder.
```
mkdir saved_ckpt  # skip if already exists

# VQVAE's checkpoint
wget -O saved_ckpt/vqvae.pth --user-agent="Mozilla/5.0" https://figshare.com/ndownloader/files/58917697

# ChargeDIFF's checkpoint
## Unconditional model
wget -O saved_ckpt/uncond.pth --user-agent="Mozilla/5.0" https://figshare.com/ndownloader/files/58917712

## Conditional models (bandgap, magnetic density, chemical system)
wget -O saved_ckpt/bandgap.pth --user-agent="Mozilla/5.0" https://figshare.com/ndownloader/files/58917700
wget -O saved_ckpt/magden.pth  --user-agent="Mozilla/5.0" https://figshare.com/ndownloader/files/58917709
wget -O saved_ckpt/chemsys.pth --user-agent="Mozilla/5.0" https://figshare.com/ndownloader/files/58917703
```

## Generate Inorganic Structures

Create a folder to store the generated structures. The following example assumes a case of storing structures in folder `./sample/test`.
```
mkdir ./sample/test

# Unconditional Generation
python generate.py --model=uncond --batch-size=50 --num-batch=10 --save-dir=./sample/test

# Conditional Generation
python generate.py --model=bandgap --batch-size=50 --num-batch=10 --save-dir=./sample/test --target=4.0
python generate.py --model=magnetic_density --batch-size=50 --num-batch=10 --save-dir=./sample/test --target=0.15

```

## Charge Density-based Inverse Design
In the jupyter notebook file named `demo_chage_density_inpainting.ipynb`, charge density-based inverse design scheme for the generation of structures with low density profile with 1D and 2D channels is described. `chemsys.pth` must be downloaded before running.


# Train ChargeDIFF

## Download the pretrained weight

First create a folder `./data` to download the Charge-MP-20 dataset.
```
mkdir data  # skip if already exists
wget -O ./data/MP-20-Charge.tar --user-agent="Mozilla/5.0" https://figshare.com/ndownloader/files/58973161
tar -xvf ./data/MP-20-Charge
```

## Start training

Run launcher files in create a folder `./launchers` to start running.
```
# unconditional model
./launchers/train_chargediff_uncond.sh

# conditional model
./launchers/train_chargediff_bandgap.sh
./launchers/train_chargediff_magnetic_density.sh
./launchers/train_chargediff_density.sh

# vqvae model (optional)
./launchers/train_vqvae.sh
```


# <a name="citation"></a> Citation

If you find this code helpful, please consider citing:

1. arxiv version
```BibTeX
@article{,
  author={Park, Junkil and Choi, Junyoung and Jung, Yousung},
  title={Integrating electronic structure into generative modeling of inorganic materials},
  Journal={arxiv},
  year={2025},
}
```

# Acknowledgement

This project was funded by National Research Foundation of Korea under grant No.RS-2024-00464386.

# License
This project is licensed under the MIT License. Please check the [LICENSE](https://github.com/parkjunkil/c/blob/main/LICENSE) file for more information.



