# RADM

This repo contains the official implementation of our ICML paper: ***Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment*** (https://arxiv.org/abs/2506.10186)

Our code is based on [GeoLDM](https://github.com/MinkaiXu/GeoLDM) and [EDM](https://github.com/ehoogeboom/e3_diffusion_for_molecules), with the following major modifications:
- A [non-equivariant autoencoder](https://github.com/skeletondyh/RADM/blob/e573c485f96ed127d763b721f999fc7ea3ac7a75/qm9/models.py#L55) used to learn rotational alignment
- [Diffusion Transformer](https://github.com/skeletondyh/RADM/blob/main/egnn/dit.py) as the noise prediction network of the diffusion model
- Other minor changes to make the new modules compatible with the existing framework

### Environment
```
pytorch==2.5.1
roma==1.5.1
rdkit
timm
numpy
scipy
wandb
imageio
```

### QM9

To train the autoencoder:

```python qm9_ae.py --n_epochs 200 --batch_size 64 --nf 256 --n_layers 9 --rot_layers 2 --lr 1e-4 --test_epochs 2 --ema_decay 0 --latent_nf 1 --dp False --exp_name qm9_ae --clip_grad False```

To train the diffusion model:

```python qm9_ldm.py --n_epochs 6000 --n_stability_samples 1000 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 256 --lr 1e-4 --test_epochs 20 --ema_decay 0.9999 --latent_nf 1 --exp_name qm9_ldm_base --dp False --size base --ae_path /path/to/ae```

where `--size` can be either `base` or `small`, and `--ae_path` should be the local path to your autoencoder checkpoint.

### Drugs

Please follow `data/geom/README.md` to prepare the data.

To train the autoencoder:

```python drugs_ae.py --n_epochs 4 --batch_size 32 --nf 256 --n_layers 4 --rot_layers 2 --lr 1e-4 --test_epochs 1 --ema_decay 0 --latent_nf 2 --clip_grad False --exp_name drugs_ae```

Note that training on Drugs requires a lot of GPU memory. If you use multiple GPUs, please change [this line](https://github.com/skeletondyh/RADM/blob/ef79321ea3ad78f00e1a59e271758a32c2253280/qm9/losses.py#L28) to: `return dist_loss.mean(), node_loss.mean()`, to deal with the aggregation.

To train the diffusion model:

```python drugs_ldm.py --n_epochs 60 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_steps 1000 --diffusion_noise_precision 1e-5 --diffusion_loss_type l2 --batch_size 256 --lr 1e-4 --test_epochs 1 --ema_decay 0.9999 --latent_nf 2 --exp_name drugs_ldm_base --size base --ae_path /path/to/ae```

### Evaluation

To analyze the sample quality of generated molecules:

```python eval_analyze.py --model_path /path/to/model --n_samples 10000```

To visualize some generated molecules：

```python eval_sample.py --model_path /path/to/model --n_samples 100```

### Conditional Generation

To train a conditional RADM:

```python qm9_ldm.py --n_epochs 4000 --n_stability_samples 500 --diffusion_noise_schedule polynomial_2 --diffusion_noise_precision 1e-5 --diffusion_steps 1000 --diffusion_loss_type l2 --batch_size 256 --lr 1e-4 --test_epochs 20 --ema_decay 0.9999 --latent_nf 1 --dp False --size base --ae_path /path/to/ae --dataset qm9_second_half --conditioning alpha --exp_name qm9_alpha```

where `--conditioning` can be one of the target properties: `alpha`, `gap`, `homo`, `lumo`, `mu`, `Cv`

To train a property predictor, `cd qm9/property_prediction` and then:

```python main_qm9_prop.py --num_workers 2 --lr 5e-4 --property alpha --exp_name exp_class_alpha --model_name egnn```

where `--property` can be one of the above target properties.

To evaluate the pretrained predictor on samples drawn from RADM:

```python eval_conditional_qm9.py --generators_path /path/to/model --classifiers_path qm9/property_prediction/outputs/exp_class_alpha --property alpha  --iterations 100  --batch_size 100 --task edm```

### Checkpoints

We provide some trained model weights [here](https://drive.google.com/drive/folders/1FwcC6sQbZj947FP33yvudEQ7SfxU4Whv?usp=drive_link). Please create a folder named `outputs` under the main folder and unzip the files into `outputs`.

### BibTeX

```
@inproceedings{ding2025radm,
  title={Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment},
  author={Ding, Yuhui and Hofmann, Thomas},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```
