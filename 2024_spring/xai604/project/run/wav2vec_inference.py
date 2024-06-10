# Standard imports
import glob
import os

# Third-party imports
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoProcessor
from torch.utils import data
import tensorflow as tf
import torch

# Custom imports
from data import speech_data_helper

db_top_dir = "/home/chanwcom/databases/"
train_top_dir = os.path.join(db_top_dir, "stop/music_train_tfrecord")
test_top_dir = os.path.join(db_top_dir,
                            "stop/test_0_music_random_300_tfrecord")

# yapf: disable
op = speech_data_helper.SpeechDataToWave()
train_dataset = tf.data.TFRecordDataset(
    glob.glob(os.path.join(train_top_dir, "*tfrecord-*")),
              compression_type="GZIP")
train_dataset = train_dataset.batch(1)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(op.process)
# yapf: enable

# yapf: disable
test_dataset = tf.data.TFRecordDataset(
    glob.glob(os.path.join(test_top_dir, "*tfrecord-*")),
              compression_type="GZIP")
test_dataset = test_dataset.batch(1)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(op.process)
# yapf: enable

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")


class IterDataset(data.IterableDataset):

    def __init__(self, tf_dataset):
        self._dataset = tf_dataset
        op = speech_data_helper.SpeechDataToWave()

    def __iter__(self):
        for data in self._dataset:
            output = {}
            output["input_values"] = [tf.squeeze(data[0]["SEQ_DATA"]).numpy()]
            output["input_length"] = tf.squeeze(data[0]["SEQ_LEN"]).numpy()
            output["labels"] = (
                data[1]["SEQ_DATA"][0].numpy().decode("unicode_escape"))

            yield (output)


pytorch_train_dataset = IterDataset(train_dataset)
pytorch_test_dataset = IterDataset(test_dataset)

transcriber = pipeline(
    "automatic-speech-recognition",
    model=
    "/home/chanwcom/local_repositories/cognitive_workflow_kit/tool/models/asr_stop_model_final/checkpoint-5000"
)

for data in pytorch_test_dataset:
    ref = data["labels"]
    hyp = transcriber(data["input_values"])
    print(f"REF: {ref}")
    print(f"HYP: {hyp}")
