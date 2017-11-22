# Efficient Architecture Search by Network Transformation

Code for the paper [Efficient Architecture Search by Network Transformation](https://arxiv.org/abs/1707.04873) in AAAI-18. 

## Dependencies

* Python 3.6 
* Tensorflow 1.3.0

## Top Nets

| nets               | test accuracy (%)       | Dataset  |
| ----------------------- | ------------- | ----- |
| [C10+_Conv_Depth_20](https://drive.google.com/open?id=1BaSHPXSTxKO5avmtzJGwinLUkSPbwJYf)     | 95.77 | C10+ |
| [C10+_DenseNet_Depth_76](https://drive.google.com/open?id=1zXTB_DmS7i9HiDAxmzrBLmwjmZmfXI2n)     | 96.56 | C10+ |
| [C10_DenseNet_Depth_70](https://drive.google.com/open?id=1T0UMowk6lN9GzDmWcjwMG6lmbh9rogXx)     | 95.34 | C10 |
| [SVHN_Conv_Depth_20](https://drive.google.com/open?id=14CoT52n6Q-dOXSHQPGNGlIh_0SjXE6q7)    | 98.27 | SVHN |

For checking these networks, please download the corresponding model files and run the following command under the folder of **code**:
```bash
$ python3 main.py --test --path=<nets path>
```

For example, by running
```bash
$ python3 main.py --test --path=../final_nets/C10+_Conv_Depth_20
```
you will get
```bash
Testing...
mean cross_entropy: 0.210500, mean accuracy: 0.957700
test performance: 0.9577
```

## Acknowledgement
The DenseNet part of this code is based on the [repository by Illarion](https://github.com/ikhlestov/vision_networks). Many thanks to [Illarion](https://github.com/ikhlestov). 

## Reference
```bash
@article{cai2017reinforcement,
  title={Reinforcement Learning for Architecture Search by Network Transformation},
  author={Cai, Han and Chen, Tianyao and Zhang, Weinan and Yu, Yong and Wang, Jun},
  journal={arXiv preprint arXiv:1707.04873},
  year={2017}
}
```
