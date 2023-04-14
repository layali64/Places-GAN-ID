# -*- coding: utf-8 -*-
"""projected_gan_Places.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AaPPqC9sfx2tMEzYkp-zP_IP2B-nZoLv

# Training Projected GAN
This is a self-contained notebook for training Projected GAN.

## Setup

Make sure you're running a GPU runtime; if not, select "GPU" as the hardware accelerator in Runtime > Change Runtime Type in the menu. 

Now, get the repo and install missing dependencies.
"""

#%cd /content/drive/MyDrive/Places/Projected-GAN

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# %%bash
# # # clone repo
# # git clone https://github.com/autonomousvision/projected_gan
# # pip install timm dill

!pip install timm dill

!pip install timm==0.5.4
!pip install ftfy
!pip install Ninja
!pip install setuptools==59.5.0
!pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f

"""## Data Preparation
We need to download and prepare the data. In this example, we use the few-shot datasets of the [FastGAN repo](https://github.com/odegeasslbc/FastGAN-pytorch).
"""

#!gdown https://drive.google.com/u/0/uc?id=1aAJCZbXNHyraJ6Mi13dSbe7pTyfPXha0&export=download

# %%capture
# #!unzip few-shot-image-datasets.zip
# !mv few-shot-images data

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Places/Projected-GAN/projected_gan

# %%bash
# python dataset_tool.py --source=/content/drive/MyDrive/Places/Dataset --dest=/content/drive/MyDrive/Places/Projected-GAN/alter-Image --resolution=256x256

"""## Training

Now that the data is prepared, we can start training!  The training loop tracks FID, but the computations seems to lead to problems in colab. Hence, it is disable by default (```metrics=[]```). The loop also generates fixed noise samples after a defined amount of ticks, eg. below ```--snap=1```.
"""

import os
import json
import re
import dnnlib

from training import training_loop
from torch_utils import training_stats
from train import init_dataset_kwargs
from metrics import metric_main

def launch_training(c, desc, outdir, rank=0):
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

    matching_dirs = [re.fullmatch(r'\d{5}' + f'-{desc}', x) for x in prev_run_dirs if re.fullmatch(r'\d{5}' + f'-{desc}', x) is not None]
    if c.restart_every > 0 and len(matching_dirs) > 0:  # expect unique desc, continue in this directory
        assert len(matching_dirs) == 1, f'Multiple directories found for resuming: {matching_dirs}'
        c.run_dir = os.path.join(outdir, matching_dirs[0].group())
    else:                     # fallback to standard
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)


    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir, exist_ok=c.restart_every > 0)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt+') as f:
        json.dump(c, f, indent=2)

    # Start training
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=False)
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    training_loop.training_loop(rank=rank, **c)

def train(**kwargs):
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=64, w_dim=128, mapping_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise ValueError('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = 2
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise ValueError('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise ValueError('--batch must be a multiple of --gpus times --batch-gpu')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise ValueError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'pg_modules.networks_stylegan2.Generator'
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        use_separable_discs = True

    elif opts.cfg == 'fastgan':
        c.G_kwargs = dnnlib.EasyDict(class_name='pg_modules.networks_fastgan.Generator', cond=opts.cond)
        c.G_opt_kwargs.lr = c.D_opt_kwargs.lr = 0.0002
        use_separable_discs = False

    # Restart.
    c.restart_every = opts.restart_every

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Projected and Multi-Scale Discriminators
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.ProjectedGANLoss')
    c.D_kwargs = dnnlib.EasyDict(
        class_name='pg_modules.discriminator.ProjectedDiscriminator',
        diffaug=True,
        interp224=(c.training_set_kwargs.resolution < 224),
        backbone_kwargs=dnnlib.EasyDict(),
    )

    c.D_kwargs.backbone_kwargs.cout = 64
    c.D_kwargs.backbone_kwargs.expand = True
    c.D_kwargs.backbone_kwargs.proj_type = 2
    c.D_kwargs.backbone_kwargs.num_discs = 4
    c.D_kwargs.backbone_kwargs.separable = use_separable_discs
    c.D_kwargs.backbone_kwargs.cond = opts.cond

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir)

# start training!

train(
    outdir='training-runs', 
    cfg='fastgan',
    data='/content/drive/MyDrive/Places/LUSN_dataset_bedroom/lsun_train_output_dir', 
    gpus=1, 
    batch=64, 
    cond=False, 
    mirror=1, 
    batch_gpu=8, 
    cbase=32768, 
    cmax=512, 
    glr=None, 
    dlr=0.002, 
    desc='', 
    metrics=[],
    kimg=10000, 
    tick=4, 
    snap=1, 
    seed=0, 
    workers=1,
    restart_every=999999,
)

"""To inspect the samples, click on the folder symbol on the left and navigate to 

```projected_gan/training-runs/YOUR_RUN```

The files ```fakesXXXXXX.png``` are the samples for a fixed noise vector at point.
"""

!python calc_metrics.py --metrics=fid50k_full --data=/content/drive/MyDrive/Places/Projected-GAN/projected_gan/training-runs/00001-fastgan-lsun_train_output_dir-gpus1-batch64-  \
--network=/content/drive/MyDrive/Places/Projected-GAN/projected_gan/training-runs/00001-fastgan-lsun_train_output_dir-gpus1-batch64-/network-snapshot.pkl

!python calc_metrics.py --help

!python calc_metrics.py --metrics=fid50k_full --network=/content/drive/MyDrive/Places/Projected-GAN/projected_gan/training-runs/00001-fastgan-lsun_train_output_dir-gpus1-batch64-/network-snapshot.pkl