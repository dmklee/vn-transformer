## Implementation of VectorNeuron-Transformer paper

[VN-Transformer Paper](https://arxiv.org/pdf/2206.04176.pdf) | [VectorNeuron Paper](https://arxiv.org/pdf/2104.12229.pdf)


### Replicating results from paper
ModelNet40 Classification
```
python train_modelnet_cls.py
```

### ToDo:
- [ ] Replicate results on ModelNet40 Classification
- [ ] Implement late-fusion model architectures (Figure 4)
- [ ] Test with non-spatial attributes


### Citations
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
    
### Acknowledgements
Many of the vector neuron modules are taken from the [VNN repo](https://github.com/FlyingGiraffe/vnn).
