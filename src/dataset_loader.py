import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def load_dataset(base_path, max_frames=50, add_motion=True):
    X, y = [], []

    for label, class_name in enumerate(["NonFight", "Fight"]):
        class_dir = os.path.join(base_path, class_name)

        for file in os.listdir(class_dir):
            if file.endswith(".npy"):
                data = np.load(os.path.join(class_dir, file))  # shape: (frames, 99)

                if add_motion:
                    # compute motion features Δx, Δy
                    dx = np.diff(data[:, 0::3], axis=0, prepend=data[0:1, 0::3])
                    dy = np.diff(data[:, 1::3], axis=0, prepend=data[0:1, 1::3])
                    data = np.concatenate([data, dx, dy], axis=1)  # shape: (frames, 165)

                X.append(data)
                y.append(label)

    X = pad_sequences(
        X,
        maxlen=max_frames,
        dtype="float32",
        padding="post",
        truncating="post"
    )

    return np.array(X), np.array(y)
