import numpy as np
from keras.models import load_model

# def predict_action(model_path,keypoints):
#     model=load_model(model_path)

#     if keypoints.shape[0]<60:
#         pad=np.zeros((60-keypoints.shape[0],keypoints.shape[1]))
#         keypoints=np.vstack((keypoints,pad))
#     else:
#         keypoints=keypoints[:60]

#     keypoints=np.expand_dims(keypoints,axis=0)
#     prob=model.predict(keypoints)[0][0]

#     return "Fight" if prob>0.5 else "NonFight",prob

# def predict_action(model_path, keypoints, max_frames=30):
#     model = load_model(model_path)

#     if keypoints.shape[0] < max_frames:
#         pad = np.zeros((max_frames - keypoints.shape[0], keypoints.shape[1]))
#         keypoints = np.vstack((keypoints, pad))
#     else:
#         keypoints = keypoints[:max_frames]

#     keypoints = np.expand_dims(keypoints, axis=0)
#     prob = model.predict(keypoints)[0][0]

#     return "Fight" if prob > 0.5 else "NonFight", prob
# src/predict.py
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def add_motion_features(sequence, max_frames=50):
    sequence = np.array(sequence)
    if sequence.ndim == 1:
        sequence = sequence.reshape(1, -1)  # 1 frame × features

    dx = np.diff(sequence[:, 0::3], axis=0, prepend=sequence[0:1, 0::3])
    dy = np.diff(sequence[:, 1::3], axis=0, prepend=sequence[0:1, 1::3])
    sequence = np.concatenate([sequence, dx, dy], axis=1)

    # pad if too short
    if sequence.shape[0] < max_frames:
        pad = np.zeros((max_frames - sequence.shape[0], sequence.shape[1]), dtype=np.float32)
        sequence = np.vstack([sequence, pad])
    else:
        sequence = sequence[:max_frames]

    return np.expand_dims(sequence, axis=0)  # add batch dimension
import numpy as np

def pad_or_trim(sequence, max_frames):
    """
    sequence: (T, F)
    returns: (max_frames, F)
    """
    T, F = sequence.shape

    if T >= max_frames:
        return sequence[:max_frames]

    pad_len = max_frames - T
    pad = np.zeros((pad_len, F), dtype=np.float32)
    return np.vstack([sequence, pad])

def predict_action(model_path, keypoints, max_frames=50):
    model = load_model(model_path)
    # keypoints shape: (frames, 165)
    keypoints = pad_or_trim(keypoints, max_frames)
    keypoints = np.expand_dims(keypoints, axis=0)  # (1, 50, 165)

    prob = model.predict(keypoints, verbose=0)[0][0]

    label = "Fight" if prob >= 0.5 else "NonFight"
    return label, float(prob)
