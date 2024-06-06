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
`conda install -c conda-forge tensorflow`

Install HuggingFace Transformers and Datasets.
`pip install transformers datasets`

SoundFile installation
`pip install soundfile`

Librosa installtion
`conda install -c conda-forge librosa`

For speech recognition evaluation
`pip install evaluate jiwer`

Reference:
https://huggingface.co/docs/datasets/v1.11.0/installation.html
