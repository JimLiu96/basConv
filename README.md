# BasConv
A graph neural network based framework to do the basket recommendation. [Our paper](https://arxiv.org/abs/2001.09900) is 
> Liu, Zhiwei, et al. "BasConv: Aggregating Heterogeneous Interactions for Basket Recommendation with Graph Convolutional Neural Network." Proceedings of the 2020 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2020.


# Data
Part of the instacart dataset is given in this repo. We have three files:
- ``trainu2b.txt``: \
  Each row is
  ``user_id, basket_id, basket_id, ...``
- ``trainb2i.txt``: \
  Each row is ``baset_id, item_id, item_id, ...``
- ``testb2i.txt``: \
  Each row is ``baset_id, item_id, item_id, ...``

The original data we use in the paper is too big to upload. You may find complete dataset of instacart [here](https://www.instacart.com/datasets/grocery-shopping-2017)

# Running
Run the code under ``./basConv/`` folder with script:
```
python basConv.py --dataset inscart_1 --regs [1e-4] --alg_type basconv --embed_size 64 --layer_size [64,64] --lr 0.0002 --save_flag 1 --pretrain -1 --batch_size 4096 --epoch 2000 --verbose 50 --node_dropout_flag 0 --mess_dropout [0.2,0.2]
```
More info regarding the arguements, please refer to the ``./basConv/parser.py`` file.
## Enviroment
Python = ``3.6``\
Tensorflow = ``1.8+`` \
``Numpy``, ``Scipy``, ``scikit-learn`` should be installed accordingly. 

# Reference
```
@inproceedings{liu2020basconv,
  title={BasConv: Aggregating Heterogeneous Interactions for Basket Recommendation with Graph Convolutional Neural Network},
  author={Liu, Zhiwei and Wan, Mengting and Guo, Stephen and Achan, Kannan and Yu, Philip S},
  booktitle={Proceedings of the 2020 SIAM International Conference on Data Mining},
  pages={64--72},
  year={2020},
  organization={SIAM}
}
```

# Acknowledgement

We reuse some part of the code in ``Neural Graph Collaborative Filtering`` <https://github.com/xiangwang1223/neural_graph_collaborative_filtering>
