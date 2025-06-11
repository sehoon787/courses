# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import glob
import os

# Third-party imports
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoProcessor
from torch.utils import data
import torch

# Custom imports
from data import speech_data_helper

db_top_dir = "/mnt/data/database"
train_top_dir = os.path.join(db_top_dir, "stop_music/music_train")
test_top_dir = os.path.join(db_top_dir, "stop_music/music_test0")

def preprocess_sample(sample: Dict) -> Dict:
    """Preprocess a single raw sample from the WebDataset.

    This function loads the waveform from the raw bytes using torchaudio,
    extracts features using the processor's feature extractor, and tokenizes
    the transcript text.

    Args:
        sample (Dict): A dictionary containing keys 'wav' (raw audio bytes)
            and 'txt' (transcript bytes).

    Returns:
        Dict: A dictionary with keys:
            - 'input_values': processed audio feature tensor.
            - 'labels': list of token IDs corresponding to the transcript.
    """
    waveform, sample_rate = torchaudio.load(io.BytesIO(sample["wav"]))
    input_values = processor.feature_extractor(
        waveform[0], sampling_rate=sample_rate
    ).input_values[0]

    text = sample["txt"].decode("utf-8")
    labels = processor.tokenizer(text).input_ids

    return {"input_values": input_values, "labels": labels}


def make_dataset(data_dir: str) -> wds.WebDataset:
    """Create a WebDataset pipeline that loads and preprocesses data shards.

    It reads all shards named 'shard-*.tar' in the given directory,
    extracts 'wav' and 'txt' entries as tuples, converts them into dictionaries,
    and applies the preprocessing function.

    Args:
        data_dir (str): Path to the directory containing dataset shards.

    Returns:
        wds.WebDataset: The prepared dataset pipeline with preprocessing.
    """
    dataset = (
        wds.WebDataset(glob.glob(os.path.join(data_dir, "shard-*.tar")))
        .to_tuple("wav", "txt")
        .map(lambda sample: {"wav": sample[0], "txt": sample[1]})
        .map(preprocess_sample)
    )
    return dataset


processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")


test_dataset = make_dataset(test_top_dir)








transcriber = pipeline(
    "automatic-speech-recognition",
    model=
    "/home/chanwcom/local_repositories/cognitive_workflow_kit/tool/models/asr_stop_model_final/checkpoint-5000"
)

for data in test_dataset:
    ref = data["labels"]
    hyp = transcriber(data["input_values"])
    print(f"REF: {ref}")
    print(f"HYP: {hyp}")
