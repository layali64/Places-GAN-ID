# -*- coding: utf-8 -*-
"""Places-Improved Diffusion.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SMQLhdspWL0YTf2TlhS41POQIUnDnbuB
"""

#!cd /content/drive/MyDrive/Places/Improve_Diffusion

#!git clone  https://github.com/openai/improved-diffusion.git

# from google.colab import files
# files.upload()

# ! pip install kaggle

# ! mkdir ~/.kaggle

#! cp kaggle.json ~/.kaggle/

#! chmod 600 ~/.kaggle/kaggle.json

#!kaggle datasets download -d mittalshubham/images256

#!unzip /content/drive/MyDrive/Places/images256.zip -d /content/drive/MyDrive/Places/Dataset/images256.zip

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Places/improved-diffusion

!pip install -e .

!pip install mpi4py

!cd /content/drive/MyDrive/Places

!git clone https://github.com/fyu/lsun.git

!cd /content/drive/MyDrive/Places

!python /content/drive/MyDrive/Places/lsun/download.py -c bedroom

!unzip /content/drive/MyDrive/Places/L/bedroom_train_lmdb.zip -d /content/drive/MyDrive/Places/L

!pip install lmdb

!python /content/drive/MyDrive/Places/improved-diffusion/datasets/lsun_bedroom.py /content/drive/MyDrive/Places/L/bedroom_train_lmdb lsun_train_output_dir

# MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
# DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
# TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/Places/improved-diffusion

!python scripts/image_train.py --data_dir /content/drive/MyDrive/Places/lsun_train_output_dir # $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS



# import torch
# torch.cuda.empty_cache()

# import gc
# del variables
# gc.collect()

#torch.cuda.memory_summary(device=None, abbreviated=False)

# #64
# MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
# DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
# TRAIN_FLAGS="--lr 2e-5 --batch_size 128"

#128
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

!python scripts/image_sample.py --model_path /content/drive/MyDrive/Places/Trained_File/lsun_uncond_100M_1200K_bs128.pt --num_sample 4 $MODEL_FLAGS $DIFFUSION_FLAGS

#checkpoint 3 - 128 
import matplotlib.pyplot as plt
import numpy as np
dfile = "/tmp/openai-2023-03-18-12-29-54-596265/samples_4x256x256x3.npz"
images = np.load(dfile)["arr_0"]
images.shape
plt.ion()
plt.figure()
plt.imshow(images[3])

#checkpoint 2 - 128 
import matplotlib.pyplot as plt
import numpy as np
dfile = "/tmp/openai-2023-03-18-10-53-43-022778/samples_4x256x256x3.npz"
images = np.load(dfile)["arr_0"]
images.shape
plt.ion()
plt.figure()
plt.imshow(images[3])

#checkpoint 2 - 64 
import matplotlib.pyplot as plt
import numpy as np
dfile = "/tmp/openai-2023-03-18-09-54-18-397239/samples_4x256x256x3.npz"
images = np.load(dfile)["arr_0"]
images.shape
plt.ion()
plt.figure()
plt.imshow(images[0])

#checkpoint 1
import matplotlib.pyplot as plt
import numpy as np
dfile = "/tmp/openai-2023-03-18-09-25-55-522787/samples_4x256x256x3.npz"
images = np.load(dfile)["arr_0"]
images.shape
plt.ion()
plt.figure()
plt.imshow(images[0])

#190K
import matplotlib.pyplot as plt
import numpy as np
dfile = "/tmp/openai-2023-03-18-09-09-51-373903/samples_4x64x64x3.npz"
images = np.load(dfile)["arr_0"]
images.shape
plt.ion()
plt.figure()
plt.imshow(images[0])

#100K
import matplotlib.pyplot as plt
import numpy as np
dfile = "/tmp/openai-2023-03-18-09-18-22-958854/samples_4x64x64x3.npz"
images = np.load(dfile)["arr_0"]
images.shape
plt.ion()
plt.figure()
plt.imshow(images[0])

#30K
import matplotlib.pyplot as plt
import numpy as np
dfile = "/tmp/openai-2023-03-12-22-29-21-116964/samples_4x64x64x3.npz"
images = np.load(dfile)["arr_0"]
images.shape
plt.ion()
plt.figure()
plt.imshow(images[0])
