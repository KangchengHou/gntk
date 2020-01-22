# Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels

This repository implements Graph Neural Tangent Kernel (infinitely wide multi-layer GNNs trained by gradient descent), described in the following paper:

Simon S. Du, Kangcheng Hou, Barnabás Póczos, Ruslan Salakhutdinov, Ruosong Wang, Keyulu Xu. Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels. NeurIPS 2019. [[arXiv]](https://arxiv.org/abs/1905.13192) [[Paper]](https://papers.nips.cc/paper/8809-graph-neural-tangent-kernel-fusing-graph-neural-networks-with-graph-kernels)


## Test run
Unzip the dataset file
```
unzip dataset.zip
```

Here we demonstrate how to use GNTK to perform classification on IMDB-BINARY dataset. We set the number of BLOCK operations to be 2, the number of MLP layers to be 2 and c_u to be 1.

Compute the GNTK gram matrix
```
mkdir out
python gram.py --dataset IMDBBINARY --num_mlp_layers 2 --num_layers 2 --scale uniform --jk 1 --out_dir out
```

Classification with kernel regression 
```
python search.py --data_dir ./out --dataset IMDBBINARY
```

Therefore we get the hyper-parameter search results at `./out/grid_search.csv`.

## Experiment for all datasets
To run the experiment described in our paper, please run `bash run_gram.sh` and `bash run_search.sh`
in order.

