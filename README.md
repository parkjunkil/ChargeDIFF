# ChargeDIFF
The first diffusion model for inorganic materials that explicitly incorporates electronic structure information

# Demo

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
