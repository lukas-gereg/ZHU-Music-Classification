import os
import io
import math
import torch
import wandb
import optuna
import pathlib
import numpy as np
from PIL import Image
import torch.nn as nn
import librosa.display
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

from data.dynamic_music_dataset import DynamicMusicDataset
from data.music_dataset import MusicDataset
from data.custom_subset import CustomSubset
from models.resnet_music_model import ResNetMusic
from models.one_d_cnn_rnn_music_model import OneDCnnRnnMusicModel
from utils.cross_validation import CrossValidation

def load_song_into_spectrograms(song_path: str | None, sound_augmentations: Compose | None = None, spectrogram_window: float = 30, y: np.ndarray = None, sr: int | float | None = None) -> list[Image]:
    if song_path is not None:
        y, sr = librosa.load(song_path, sr=None)

    if sound_augmentations is not None:
        y = sound_augmentations(samples=y, sample_rate=sr)

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

def generate_dataset_spectrograms(audio_dataset_path, export_path: str | None = None, sound_augmentations = None) -> None:

    path = pathlib.Path(audio_dataset_path)

    if export_path is None:
        export_path = pathlib.Path(audio_dataset_path).parent / 'generated_images'

    for folder in path.iterdir():
        folder_path = export_path / folder.name
        os.makedirs(folder_path, exist_ok=True)

        for file in folder.iterdir():
            if file.name == 'jazz.00054.wav':
                continue

            print(f"Generating spectrogram for {file.name}")
            spectrograms = load_song_into_spectrograms(file, sound_augmentations)
            spectrograms[0].save(f"{folder_path / file.stem}.png")

def objective(trial: optuna.Trial, random_seed, dataset_path) -> float:
    image_size_square = trial.suggest_int("image_size", 224, 512)
    cnn_sizes_count = trial.suggest_int("cnn_sizes_count", 1, 5)
    cnn_sizes = [trial.suggest_int(f"cnn_size_{i}", 32, 512) for i in range(cnn_sizes_count)]
    gru_layers_count = trial.suggest_int("gru_layers_count", 1, 3)
    hidden_size = trial.suggest_int("hidden_size", 128, 512)
    fc_count = trial.suggest_int("fc_count", 0, 3)
    fc = [trial.suggest_int(f"fc_{i}", 64, 512) for i in range(fc_count)]
    drop_chance = trial.suggest_float("drop_chance", 0.0, 0.6)

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
    optim_weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2)

    color_channels = 3
    model_debug = False
    folds = 5
    early_stopping = 20
    batch = 4
    scheduler_patience = 3
    scheduler_factor = 0.5

    model_params = {'color_channels': color_channels,
                    'image_size': (image_size_square, image_size_square), 'cnn_sizes': cnn_sizes, 'hidden_size': hidden_size, 'num_layers': gru_layers_count,
                    'fc_size': fc, 'drop_chance': drop_chance}

    return np.mean(run(random_seed, model_debug, early_stopping, batch, color_channels, learning_rate, optim_weight_decay, folds, model_params, dataset_path))

def run(random_seed, debug, early_stopping, batch, color_channels, lr, weight_decay, folds, model_params, dataset_path):
    epochs = 10000


    sound_augmentation = Compose([AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.3),
                       TimeStretch(min_rate=0.95, max_rate=1.05, leave_length_unchanged=True, p=0.3),
                       PitchShift(min_semitones=-1, max_semitones=1, p=0.3)])

    item_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                         transforms.Resize(model_params["image_size"], antialias=True)])

    # base_dataset = MusicDataset(dataset_path,
    #                             item_transform=item_transform,
    #                             color_channels=color_channels)
    base_dataset = DynamicMusicDataset(dataset_path,
                                       sound_augmentation,
                                       lambda y, sr, augmentation: load_song_into_spectrograms(None, y=y, sr=sr, sound_augmentations=augmentation)[0],
                                       item_transform=item_transform)

    x, labels = zip(*[item for item in base_dataset.data])
    y_ids = [i for i in range(len(base_dataset))]

    train_ids, test_ids, train_y, test_y = train_test_split(y_ids, labels, stratify=labels, test_size=0.3,
                                                            random_state=random_seed)
    train_ids, validation_ids = train_test_split(train_ids, stratify=train_y, test_size=0.3, random_state=random_seed)

    train_dataset = CustomSubset(base_dataset, train_ids)
    test_dataset = CustomSubset(base_dataset, test_ids)
    validation_dataset = CustomSubset(base_dataset, validation_ids)

    model_params["num_classes"] = len(base_dataset.classes)

    model = ResNetMusic(model_params)
    # model = OneDCnnRnnMusicModel(model_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=scheduler_patience,
    #                                            factor=scheduler_factor)
    loss = nn.CrossEntropyLoss()

    wandb_config = dict(project="ZHU-Music-Classification", entity="ZHU-Music-Classification", config={
        "model properties": model_params,
        "learning rate": lr,
        "image_transforms": str(item_transform),
        "epochs": epochs,
        "early stopping": early_stopping,
        "model": str(model),
        "optimizer": str(optimizer),
        "loss calculator": str(loss),
        "LR reduce scheduler": str(scheduler),
        "debug": debug,
        "batch_size": batch,
        "random_seed": random_seed,
        "data_augmentation": str(item_transform),
        "music_augmentation": str(sound_augmentation),
        "weight_decay": weight_decay,
    })

    wandb_login_key = "a9f105e8b3bc98e07700e93201d4b02c1c75106d"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wandb_login_key is not None:
        wandb.login(key=wandb_login_key)

    results = CrossValidation(batch, folds, debug, random_seed)(epochs, device, optimizer, model, loss,
                                                           train_dataset, validation_dataset, test_dataset,
                                                           early_stopping, scheduler, wandb_config)

    print(f"training of model complete")

    return results


if __name__ == '__main__':
    seed = 42
    study_name = "ResnetMusicClassification"
    dataset_sound_path = os.path.join('.', 'Data', 'genres_original')
    generated_dataset_path = os.path.join('.', 'Data', 'generated_images_augmented')
    # generated_dataset_path = os.path.join('.', 'Data', 'generated_images')
    # generate_dataset_spectrograms(dataset_sound_path, generated_dataset_path, augment)
    # item_path = 'C:/Users/lukiq/Downloads/Test.mp3'
    # spectrograms = load_song_into_spectrograms(item_path)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=seed), load_if_exists=True, study_name=study_name)
    study.optimize(lambda trial: objective(trial, seed, dataset_sound_path), n_trials=100)