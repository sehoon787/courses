# pylint: disable=import-error, no-member

from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = "Chanwoo Kim (chanwcom@gmail.com)"

# Standard library imports
import os

# Third-party library imports
from transformers import pipeline

# Local application imports
import sample_util

# Defines the path to the root directory containing the dataset
db_top_dir = "/mnt/data/database"

# Defines the path to the test dataset
test_top_dir = os.path.join(db_top_dir, "stop_music/music_test0")

# Load the test dataset using a custom utility function.
# The second argument (False) disables tokenization inside the dataset.
test_dataset = sample_util.make_dataset(test_top_dir, do_tokenize=False)

# Initializes the ASR pipeline with a fine-tuned checkpoint.
# This loads a model for automatic speech recognition (ASR).
transcriber = pipeline(
    "automatic-speech-recognition",
    model=(
        "/home/chanwcom/local_repositories/cognitive_workflow_kit/tool/models/"
        "asr_stop_model_final/checkpoint-2000"
    )
)

# Iterate through each sample in the dataset and decode the audio
for data in test_dataset:
    # Ground-truth reference text (original transcript).
    ref = data["labels"]

    # Run the ASR model to get the hypothesis transcription
    hyp = transcriber(data["input_values"])

    # Print the reference and hypothesis for comparison
    print(f"REF: {ref}")
    print(f"HYP: {hyp['text']}")
