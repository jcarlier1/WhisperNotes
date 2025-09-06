from faster_whisper import WhisperModel
import torch
faster_whisper_model = 'nyrahealth/faster_CrisperWhisper'

# Initialize the Whisper model

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = "float16" if torch.cuda.is_available() else "float32"
model = WhisperModel(faster_whisper_model, device=device, compute_type="float16")
audio_path = "audios/Lecture 1 EvoPsy_converted_60_120.mp3"
segments, info = model.transcribe(audio_path, beam_size=1, language='en', word_timestamps = False, without_timestamps= True)

for segment in segments:
    print(segment)