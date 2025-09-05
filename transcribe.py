import argparse
import os
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def transcribe_audio(file_path):
    print("[DEBUG] Checking device...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[DEBUG] Using device: {device}, dtype: {torch_dtype}")
    model_id = "nyrahealth/CrisperWhisper"  # You can change this to a different model if needed

    print("[DEBUG] Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="eager"
    )
    model.to(device)
    print("[DEBUG] Model loaded and moved to device.")

    print("[DEBUG] Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    print("[DEBUG] Processor loaded.")

    print("[DEBUG] Creating pipeline...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps="word",
        torch_dtype=torch_dtype,
        device=device,
    )
    print("[DEBUG] Pipeline created.")

    print(f"[DEBUG] Transcribing file: {file_path}")
    result = pipe(file_path)
    print("[DEBUG] Transcription complete.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Transcribe an audio file.")
    parser.add_argument("--f", type=str, required=True, help="Path to the audio file")
    args = parser.parse_args()

    if not os.path.exists(args.f):
        print(f"Error: The file '{args.f}' does not exist.")
        sys.exit(1)

    try:
        transcription = transcribe_audio(args.f)
        print("Transcription:")
        print(transcription["text"])
    except Exception as e:
        import traceback
        print(f"An error occurred while transcribing the audio: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
