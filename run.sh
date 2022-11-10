#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --partition=debug
#SBATCH --time=1-0:0:0
#SBATCH --account=siqiouyang
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt

# export DATA_PATH=/mnt/data/siqiouyang/datasets/

# python -m train pipeline=cifar_dev model=s4 +wandb.name=cifar_nplr
# python -m train pipeline=cifar_dev model=s4_dev +wandb.name=cifar_nplr_alpha

# python -m train experiment=sc/s4-sc model=s4 +wandb.name=sc_nplr
# python -m train experiment=sc/s4-sc model=s4_freeze +wandb.name=sc_nplr_freeze
# python -m train experiment=sc/s4-sc model=s4_dev +wandb.name=sc_nplr_alpha

# export DATA_PATH=/mnt/data/siqiouyang/datasets/lra_release 
# python -m train experiment=lra/s4-lra-pathx model=s4 +wandb.name=pathx_nplr
# python -m train experiment=lra/s4-lra-pathx-freeze model=s4 +wandb.name=pathx_nplr_freeze
# python -m train experiment=lra/s4-lra-pathx-freeze-all model=s4 +wandb.name=pathx_nplr_freeze_all
# python -m train experiment=lra/s4-lra-pathx model=s4_dev +wandb.name=pathx_nplr_alpha

# repeat
# python -m train pipeline=repeat model=s4_dev +wandb.name=repeat_s4_posenc
# python -m train pipeline=repeat model=liquid_s4_dev dataset.n_train=10000 +wandb.name=repeat_liquids4_posenc_10xtrain
# python -m train pipeline=repeat model=liquid_s4_dev dataset.n_train=10000 +wandb.name=repeat_liquids4_large_posenc_10xtrain
# python -m train pipeline=repeat model=transformer +wandb.name=repeat_transformer
# python -m train pipeline=repeat dataset.n_train=10000 trainer.max_epochs=1000 model=mega_dev model.layer.0.disable_ema=True model.layer.0.attn_dim_qk=8 model.layer.0.attn_dim_value=32 +wandb.name=repeat_mega_w/o_ema_10xtrain
# python -m train pipeline=repeat dataset.n_train=10000 trainer.max_epochs=1000 model=mega_dev model.layer.0.ema_heads=1 model.layer.0.attn_dim_qk=8 model.layer.0.attn_dim_value=32 +wandb.name=repeat_mega_10xtrain

# repeatreg
# python -m train pipeline=repeatreg model=transformer +wandb.name=repeatreg_transformer

# parity