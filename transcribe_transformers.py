import os
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from utils import adjust_pauses_for_hf_pipeline_output


# "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"
# torch.float16 if torch.cuda.is_available() else
torch_dtype = torch.float32

model_id = "nyrahealth/CrisperWhisper"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps='word',
    torch_dtype=torch_dtype,
    device=device,
)

audio_path = "audios/Lecture 1 EvoPsy_converted_60_120.mp3"
hf_pipeline_output = pipe(audio_path)
crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)
print(crisper_whisper_result)