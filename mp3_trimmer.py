import sys
import os
from pydub import AudioSegment

def crop_mp3(input_path, start_sec, end_sec):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".m4a":
        # Convert m4a to mp3 first
        audio = AudioSegment.from_file(input_path, format="m4a")
        mp3_path = os.path.splitext(input_path)[0] + "_converted.mp3"
        audio.export(mp3_path, format="mp3")
        input_path = mp3_path
        print(f"Converted m4a to mp3: {mp3_path}")
    # Load MP3
    audio = AudioSegment.from_mp3(input_path)
    # Crop audio
    cropped = audio[start_sec * 1000:end_sec * 1000]
    # Prepare output filename
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = f"audios/{base}_{start_sec}_{end_sec}.mp3"
    # Export cropped audio
    cropped.export(output_path, format="mp3")
    print(f"Cropped file saved as: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python mp3_trimmer.py <input_mp3> <start_sec> <end_sec>")
        sys.exit(1)
    input_mp3 = sys.argv[1]
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    crop_mp3(input_mp3, start, end)