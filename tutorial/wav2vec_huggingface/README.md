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


2. Sentence Piece

https://pypi.org/project/sentencepiece/
`pip install sentencepiece`


```spm_train --input=train-all.trans_without_uttid.txt \
       --model_prefix=model_unigram_${vocab_size}  \
       --vocab_size=${vocab_size} \
       --character_coverage=1.0 \
       --model_type=unigram```
spm_train --input=train-all.trans_without_uttid.txt \
       --model_prefix=model_bpe_${vocab_size}  \
       --vocab_size=${vocab_size} \
       --character_coverage=1.0 \
       --model_type=bpe```

