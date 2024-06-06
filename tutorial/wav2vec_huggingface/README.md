# 1. Setup the environment.


Create the Conda environment.

`conda create --name py3_10_hf python=3.10`

Activate the Conda environment created just ago.

`conda activate py3_10_hf`


Install Pytorch and Tensorflow.

https://pytorch.org/get-started/locally/
Check the CUDA version
`nvcc --version`

Select the command at the bottom of the table, after seleting the right "Compute Platform"
For example, if the CUDA version is 11.8, then run the following command:
Note that torchdata is added.


`conda install pytorch torchvision torchaudio torchdata pytorch-cuda=11.8 -c pytorch -c nvidia``
\
`conda install tensorflow`




Install HuggingFace Transformers.

`conda install conda-forge::transformers`

Install HuggingFace Datasets.

`conda install -c huggingface -c conda-forge datasets`

SoundFile installation
`pip install soundfile`

Librosa installtion
`conda install librosa`

Reference:
https://huggingface.co/docs/datasets/v1.11.0/installation.html
