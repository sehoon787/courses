from datasets import load_dataset, Audio
from transformers import pipeline

dataset = load_dataset("PolyAI/minds14", "en-US", split="train[:100]")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
sampling_rate = dataset.features["audio"].sampling_rate
audio_file = dataset[0]["audio"]["path"]

transcriber = pipeline("automatic-speech-recognition",
                       model="./asr_mind_model01/checkpoint-4000")
#transcriber = pipeline("automatic-speech-recognition", model="stevhliu/my_awesome_asr_mind_model")
print(transcriber(audio_file))
