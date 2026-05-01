# Optimistic $\epsilon$-Greedy Exploration for Cooperative Multi-Agent Reinforcement Learning

This repository is the official implementation of the paper: **"Optimistic $\epsilon$-Greedy Exploration for Cooperative Multi-Agent Reinforcement Learning"**. 

Our code is built upon the open-source [WQMIX](https://github.com/oxwhirl/wqmix) repository.

## 1. Preparation

### Dependencies
The dependencies required for this repository are almost identical to those of [WQMIX](https://github.com/oxwhirl/wqmix). Please refer to the `requirements.txt` file for specific version numbers. The experimental results reported in our paper were obtained using these specific versions.

### Environments
- **Matrix Games** and **Pred and Prey**: These environments are built into the repository.
- **StarCraft II (SMAC)**: If you wish to run experiments on SMAC, please follow the installation instructions provided by [PyMARL](https://github.com/oxwhirl/pymarl).

## 2. Running Experiments

You can start an experiment using the following command:

```
python main.py --config=opt_qmix --env-config=sc2 with env_args.map_name=5m_vs_6m
```

`--config`: Refers to the algorithm configuration files in config/algs.

`--env-config`: Refers to the environment configuration files in config/envs.

**Important Note: Parameter Adjustments**

Please note that the default parameters in `config/algs` are specifically tuned for the SMAC environment. If you wish to run *opt_qmix* or *opt_vdn* on MatrixGames or Pred and Prey, please modify the parameters in the corresponding configuration files to achieve the expected performance:

**For all non-SMAC environments:**

Set `w` to `0.01`.

**For Matrix Games:**

Set `buffer_size` to `32`

Set `target_update_interval` to `1`

Set `use_temp_norm` to `False`.

Set `use_norm` to `False`.

**For Pred and Prey:**

Set `epsilon_anneal_time` to `200000`.

Set `normalmax` to `2`.

Set `normal_anneal_time` to `20000`.

**Ablation Studies**

Noise-based Ablation: Set `action_selector` to `"noise_greedy"`.

Intrinsic Reward Ablation: Set `action_selector` to `"epsilon_greedy"` AND set `use_internal` to `True`.

## 3. Acknowledgements

During the development of this project, we referred to parts of the implementation from [PyMARL](https://github.com/oxwhirl/pymarl) and [EPyMARL](https://github.com/uoe-agents/epymarl). We would like to express our sincere gratitude to the authors of these open-source frameworks.

## 4. Citation
If you find this code or research helpful, please cite our paper:

```
@article{zhang2025optimistic,
  title={Optimistic {$\epsilon$}-Greedy Exploration for Cooperative Multi-Agent Reinforcement Learning},
  author={Zhang, Ruoning and Wang, Siying and Chen, Wenyu and Zhou, Yang and Zhao, Zhitong and Zhang, Zixuan and Zhang, Ruijie and Albrecht, Stefano V.},
  journal={arXiv preprint arXiv:2502.03506},
  year={2025}
}
```