import os
import io
import math
import torch
import wandb
import pathlib
import numpy as np
from PIL import Image
import torch.nn as nn
import librosa.display
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split


from assignment2.data.custom_subset import CustomSubset
from assignment2.utils.cross_validation import CrossValidation

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
    debug = False
    folds = 5
    random_seed = 42

    scheduler = None
    early_stopping = 25

    BATCH_SIZE = 64
    image_size = (150, 150)
    color_channels = 3

    epochs = 10000
    lr = 0.0003

    item_transform = transforms.Compose([transforms.ToTensor(),
                                         # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         transforms.Resize(image_size, antialias=True)])
    base_dataset = Covid19Dataset(os.path.join(".", "COVID-19_Radiography_Dataset"),
                                  color_channels=color_channels,
                                  item_transform=item_transform)

    x, y = zip(*[item for item in base_dataset.data])
    y_ids = [i for i in range(len(base_dataset))]

    train_ids, test_ids, train_y, test_y = train_test_split(y_ids, y, stratify=y, test_size=0.3,
                                                            random_state=random_seed)
    train_ids, validation_ids = train_test_split(train_ids, stratify=train_y, test_size=0.3, random_state=random_seed)

    train_dataset = CustomSubset(base_dataset, train_ids)
    test_dataset = CustomSubset(base_dataset, test_ids)
    validation_dataset = CustomSubset(base_dataset, validation_ids)

    model_properties = {'color_channels': color_channels, 'image_size': image_size,
                        'pooling_method_constructor': nn.AdaptiveAvgPool2d}
    model = CombinedVVGModel(model_properties)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    wandb_config = dict(project="ZHU-Music-Classification", entity="ZHU-Music-Classification", config={
        "model properties": model_properties,
        "learning rate": lr,
        "image_transforms": str(item_transform),
        "epochs": epochs,
        "early stopping": early_stopping,
        "model": str(model),
        "optimizer": str(optimizer),
        "loss calculator": str(loss),
        "LR reduce scheduler": str(scheduler),
        "debug": debug,
        "batch_size": 64,
        "random_seed": random_seed
    })

    wandb_login_key = "a9f105e8b3bc98e07700e93201d4b02c1c75106d"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wandb_login_key is not None:
        wandb.login(key=wandb_login_key)

    CrossValidation(BATCH_SIZE, folds, debug, random_seed)(epochs, device, optimizer, model, loss,
                                                           train_dataset, validation_dataset, test_dataset,
                                                           early_stopping, scheduler, wandb_config)

    print(f"training of model complete")