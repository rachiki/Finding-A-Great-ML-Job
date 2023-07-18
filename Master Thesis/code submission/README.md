# General
The majority of the experiments are contained in the notebook "Experiments-MNIST,CIFAR,NEWS,PTB_XL.ipynb".
The other experiments on Tiny ImagNet were run with slurm utilizing the resources of the chair. Those are contained in the imagenet_code folder.

We also use PyTorch and Torchvision throughout the experiments:

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., … Chintala, S. (2019). 
PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (pp. 8024–8035). 
Curran Associates, Inc. Retrieved from http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf

Marcel, S., & Rodriguez, Y. (2010, October). Torchvision the machine-vision package of torch. In Proceedings of the 18th ACM international conference on Multimedia (pp. 1485-1488).


# Notebook:
Experiments for MNIST, CIFAR10 and 20Newsgroups are combined as that was those were the development datasets for the thesis.

The structure is as follows.
0. is for library imports and to show the available GPU.     
1. contains all MNIST, CIFAR10 and 20Newsgroups experiments.
1.1 contains model definitions
1.2 contains dataset definitions
1.3 contains training loop definitions
1.4 contains all experiments where models are trained with attribution based augmentation or some other methods
1.5 contains the experiments with frozen models
1.6 contains illustrations for the attribution selection step and augmentation step
2. contains the PTB-XL experiment

Guide for use:
NOTE: datasets and glove embeddings are not included, download of CIFAR10 and MNIST is automated, 20Newsgroups and the glove embeddings, as well as PTB-XL need to be downloaded.
Locations for download are described in the notebook.

Sections 0 to 1.3 can all be run instantly as they are just definitions.
Section 1.4 has a wide variety of settings that are explained in the section. 
	Tables 3.1, 3.2, 4.2, A.1 and A.2 are derived from this part.
Section 1.5 uses the last model checkpoint, freezes the model and creates the graphs.
	Figure 3.3 and A.1 was created with this part.
Section 1.6 illustrates the attribution selection step and augmentation step. 
	Figure 3.2 was created with this part.
Section 2 with the PTB-XL experiment can be run seperately from Section 1. the exact process is explained in the notebook.
	Table 4.4 and A.4 are derived from this part.

Sources for datasets, models and code snippets are cited throughout the notebook.


# Tiny ImageNet
This is based on https://github.com/pytorch/examples/tree/main/imagenet.
We did not download Tiny ImageNet as it is at the chair server, but it can be found at https://huggingface.co/datasets/Maysee/tiny-imagenet for example.
	Table 4.3 and A.3 are derived from this part.	

One only has to run sbatch train.sbatch
The settings in train.sbatch are similar to the notebook with the exception of: 
--augment being there for toggling RandAugment and
--tinytransformer being set to 1 to toggle the vision transformer settings with patch_size=8