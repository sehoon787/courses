# 1. Setup the environment.


Create the Conda environment.

`conda create --name py3_10_hf python=3.10`

Activate the Conda environment created just ago.

`conda activate py3_10_hf`


Install Pytorch and Tensorflow.


`conda install pytorch torchvision torchaudio torchdata -c pytorch`

`conda install tensorflow`

Install HuggingFace Transformers.

`conda install conda-forge::transformers`

Install HuggingFace Datasets.

`conda install datasets`
