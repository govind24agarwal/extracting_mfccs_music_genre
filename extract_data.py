import json
import os
import math
import librosa

DATASET_PATH = "/home/govind/Documents/ML/Velario Youtube/extracting_mfccs_music_genre/genres_original"
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Generating mfcc of every segment of every music clip of ever
        grnre and storing it into json file along with it's labels.

    Args:
        dataset_path (String): path of dataset of music clips
        json_path (String)): path of json file  where mfccs should be saved
        num_mfcc (int, optional): Number of mfcc coefficients to consider. Defaults to 13.
        n_fft (int, optional): Number of samples per frame. Defaults to 2048.
        hop_length (int, optional): Hop length for mfcc calculation. Defaults to 512.
        num_segments (int, optional): Number of segments in each music clip. Defaults to 5.
    """
    data = {
        "distinct_genres": [],
        "labels": [],
        "mfcc": []
    }
    samples_per_segment = int(SAMPLES_PER_TRACK/num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment/hop_length)

    # Reccursively going through every folder in dataset_path
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # Going through folder of each genre
        if dirpath is not dataset_path:
            semantic_label = dirpath.split("/")[-1]
            data["distinct_genres"].append(semantic_label)
            print("Processing for genre: {}".format(dirpath))

            # Going through every music clip in it's genre's  folder
            for f in filenames:
                try:
                    file_path = os.path.join(dirpath, f)
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                    # Dividing file into segments
                    for d in range(num_segments):
                        start = samples_per_segment * d
                        end = start + samples_per_segment

                        mfcc = librosa.feature.mfcc(
                            signal[start:end], sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                        mfcc = mfcc.T

                        # Saving mfccs
                        if len(mfcc) == num_mfcc_vectors_per_segment:
                            data["mfcc"].append(mfcc.tolist())
                            data["labels"].append(i-1)
                            print("{}, segment: {}".format(f, d+1))
                except Exception as e:
                    print(e)

    # Writing all data collected into json file
    with open(json_path, "w") as fj:
        json.dump(data, fj, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
