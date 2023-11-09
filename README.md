# Implementation of VectorNeuron-Transformer paper (wip)

[VN-Transformer Paper](https://arxiv.org/pdf/2206.04176.pdf) | [VectorNeuron Paper](https://arxiv.org/pdf/2104.12229.pdf)


## Running Experiments from Paper
### ModelNet40 Classification
Download the data from [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip).  Run the following
script to train the model (it takes around 1 min per epoch on NVIDIA 2080).
```
python train_modelnet_cls.py --data_path=path/to/modelnet40_normal_resampled
```
So far, I have not been able to replicate the results from the paper.  Currently, running a hyperparameter search based on Table 5.  The best I have gotten is 82.7% test acc after 1000 epochs (32hidden, 4heads, No latent).


## Citations
If you use this repo, please consider citing the original works:
```bibtex
    @article{assaad2022vn,
      title={VN-Transformer: Rotation-Equivariant Attention for Vector Neurons},
      author={Assaad, Serge and Downey, Carlton and Al-Rfou, Rami and Nayakanti, Nigamaa and Sapp, Ben},
      journal={arXiv preprint arXiv:2206.04176},
      year={2022}
    }
```

```bibtex
    @article{deng2021vn,
      title={Vector Neurons: a general framework for SO(3)-equivariant networks},
      author={Deng, Congyue and Litany, Or and Duan, Yueqi and Poulenard, Adrien and Tagliasacchi, Andrea and Guibas, Leonidas},
      journal={arXiv preprint arXiv:2104.12229},
      year={2021}
    } 
 ```
    
## Acknowledgements
Many of the vector neuron modules are taken from the [VNN repo](https://github.com/FlyingGiraffe/vnn).

## ToDo:
- [ ] Replicate results on ModelNet40 Classification
- [ ] Implement late-fusion model architectures (Figure 4)
- [ ] Test with non-spatial attributes
