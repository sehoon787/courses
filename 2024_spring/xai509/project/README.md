
# Projects goal.

Implement the CTC loss using PyTorch.
The project consists of two phases.


Due Date: By Next Weekend (June 23th).

## 1) Phase-I "Implementation of the CTC loss"
It is fine to do only the phase-I of the project, although extra (15%) points will be given if a student successfully completes the phase-II.
I wrote the Tensorflow version in https://github.com/chanwcom/cognitive_workflow_kit/tree/main/loss/Tensorflow

### a) First, I recommend you check the Tensorflow example in the following steps


Create the Conda environment.

`conda create --name py3_10_hf python=3.10`

Activate the Conda environment created just ago.

`conda activate py3_10_hf`

Install Tensorflow and Tensorflow probability.
`conda install Tensorflow Tensorflow-probability`

Download the code in https://github.com/chanwcom/cognitive_workflow_kit/tree/main/loss/Tensorflow.
Instead of downloading the entire repository, you may only download "seq_loss_util.py" and "seq_loss_util_test.py"

Run the unit test to check whether the code is executed correctly:
`python seq_loss_util_test.py`


### b) Implement the code in PyTorch.

The main entry point of the Tensorflow code in "seq_loss_util.py" is `def ctc_loss(labels, labels_len, logits, logits_len)`.

First, implement "to_blank_augmented_labels" in PyTorch.

Second, implement "label_trans_table" in PyTorch.

Third, implement "calculate_alpha_beta" in PyTorch. "calculate_alpha_beta" will call "_calculate_unnormalized_log_seq_prob" and "calculate_log_label_prob".

Finally, implement the entire "ctc_loss" method.


I will create a PyTorch unit test which is doing the same thing as "seq_loss_util_test.py" to check whether your code passes the unit test.










## 2) Phase-II
Fine tune the model using the implemented CTC loss.


# 1. Setup the environment.


Create the Conda environment.

`conda create --name py3_10_hf python=3.10`

Activate the Conda environment created just ago.

`conda activate py3_10_hf`


Install PyTorch and Tensorflow.

https://pytorch.org/get-started/locally/
Check the CUDA version
`nvcc --version`

Select the command at the bottom of the table, after seleting the right "Compute Platform"
For example, if the CUDA version is 11.8, then run the following command:
Note that torchdata is added.


`conda install pytorch torchvision torchaudio torchdata pytorch-cuda=11.8 -c pytorch -c nvidia``
\
`conda install Tensorflow`

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

# 2. Bazel installation

Use bazelisk
https://bazel.build/install/ubuntu


# 3. STOP dataset

We used the music portion.

But we removed 00011525.wav.

So the total number of utterances is reduced from 11563 to 11562.

# 4. Run the scripts in the "run" directory
