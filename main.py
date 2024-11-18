import io
import math
import os
import pathlib
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_song_into_spectrograms(song_path: str, spectrogram_window: float = 30) -> list[Image]:
    y, sr = librosa.load(song_path, sr=None)
    window_samples = int(sr * spectrogram_window)
    spectrograms = []

    for spectrogram_idx in range(math.ceil(len(y) / window_samples)):
        start_sample = spectrogram_idx * window_samples
        end_sample = min((spectrogram_idx + 1) * window_samples, len(y))

        if not math.isclose((end_sample - start_sample) / window_samples, 1, rel_tol=0.005):
            continue

        audio_segment = y[start_sample:end_sample]

        mel_spectrogram = librosa.feature.melspectrogram(y=audio_segment, sr=sr, n_fft=2048, hop_length=512)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        fig, ax = plt.subplots()
        librosa.display.specshow(mel_spectrogram_db, hop_length=512, x_axis='time', y_axis='mel')

        buf = io.BytesIO()

        ax.set_axis_off()
        fig.savefig(buf, bbox_inches='tight', pad_inches=0.0)

        buf.seek(0)
        im1 = Image.open(buf).convert("RGB")

        buf.close()
        plt.close(fig)

        spectrograms.append(im1)

    return spectrograms

def generate_dataset_spectrograms(audio_dataset_path):
    path = pathlib.Path(audio_dataset_path)
    output_path = pathlib.Path(audio_dataset_path).parent / 'generated_images'

    for folder in path.iterdir():
        folder_path = output_path / folder.name
        os.makedirs(folder_path, exist_ok=True)

        for file in folder.iterdir():
            if file.name == 'jazz.00054.wav':
                continue

            print(f"Generating spectrogram for {file.name}")
            spectrograms = load_song_into_spectrograms(file)
            spectrograms[0].save(f"{folder_path / file.stem}.png")

if __name__ == '__main__':
    dataset_path = './archive/Data/genres_original'
    generate_dataset_spectrograms(dataset_path)
    # item_path = 'C:/Users/lukiq/Downloads/Test.mp3'
    # spectrograms = load_song_into_spectrograms(item_path)
    print("test")