
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

The following is the recommended steps to impelement the `ctc_loss` method in PyTorch.

- First, implement the `to_blank_augmented_labels` method in PyTorch.

- Second, implement `label_trans_table` in PyTorch.

- Third, implement `calculate_alpha_beta` in PyTorch. `calculate_alpha_beta` will call `_calculate_unnormalized_log_seq_prob` and `calculate_log_label_prob`.

- Finally, implement the entire `ctc_loss` method.


I will create a PyTorch unit test which is doing the same thing as "seq_loss_util_test.py" to check whether your code passes the unit test.



## 2) Phase-II
Phase-II is the optional phase, but students will the get extra 15% of points if he or she completes this phase.
You will use the toolkit used for XAI 604 project "https://github.com/chanwcom/courses/tree/main/2024_spring/xai604/project"

But unlike 604 project, you will only use the speech recogniton part.

- First, set up the environment mentioned in https://github.com/chanwcom/courses/blob/main/2024_spring/xai604/project/README.md.

- Second, perform fine tuning using `wav2vec_finetuing.py` in https://github.com/chanwcom/courses/tree/main/2024_spring/xai604/project/run.
  
- Third, perform inferencing using `wav2vec_inference.py` in the same directory.

 The objective is using your loss function created in Phase-I instead of the default CTC loss.
 You may use the following approach to use your loss function.
```
from transformers import Trainer

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        ctc_loss = ... your ctc loss ...
        return ctc_loss
```
