import os
import io
import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ================= Global Seed =================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================= Configuration =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POS_DATA_DIR = os.path.join(BASE_DIR, "data_samples", "4_train_5s")
NEG_HUMAN_DIR = os.path.join(BASE_DIR, "data_samples", "3_human_daily")
NEG_SHAKE_DIR = os.path.join(BASE_DIR, "data_samples", "2_shaking_table")

ORIGINAL_MODEL_PATH = os.path.join(BASE_DIR, "train_model", "fold_0.h5")
SAVE_FINETUNED_PATH = os.path.join(BASE_DIR, "fine_tune_model", "finetuned_v1.h5")

FS = 100.0
WIN_PTS = 500          # 5 s window at 100 Hz
BUFFER_PTS = 1000      # 10 s physical buffer (fallback when only 1 neg file exists)
LEARNING_RATE = 1e-4
EPOCHS = 100
BATCH_SIZE = 32


# ================= Helper Functions =================
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss


def generate_gaussian_label(length, p_idx, sigma=10):
    x = np.arange(length)
    y = np.exp(-((x - p_idx) ** 2) / (2 * sigma ** 2))
    return y.reshape(-1, 1)


def extract_event_id(fname):
    """
    Unified event ID extraction from filename (compatible with both
    legacy and newer naming formats) to enforce event-level splitting.
    """
    parts = fname.split('_')
    first = parts[0]
    if len(first) >= 10:
        return first[6:]           # e.g., AIC0052102180358 -> 2102180358
    elif len(parts) >= 3:
        return parts[2][2:]        # e.g., YN_LH001_YN202504... -> 202504...
    return first


def clean_and_load_txt(fpath):
    """Robustly load triaxial smartphone acceleration .txt files."""
    try:
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            clean_lines = [line for line in f if line.strip() and line.strip()[0].isdigit()]
        if len(clean_lines) < WIN_PTS:
            return None
        df = pd.read_csv(io.StringIO('\n'.join(clean_lines)), sep=r'\s+', header=None, engine='c')
        if df.empty or df.shape[1] < 3:
            return None
        return df.iloc[:, -3:].values.astype(np.float32)
    except Exception:
        return None


def normalize_clip(clip):
    clip_demean = clip - np.mean(clip, axis=0)
    max_v = np.max(np.abs(clip_demean))
    return clip_demean / max_v if max_v > 1e-6 else clip_demean


# ================= Data Preparation (Leakage-Free) =================
def load_pos_from_files(file_list):
    X, Yc, Yp = [], [], []
    for fpath in file_list:
        fname = os.path.basename(fpath)
        try:
            p_idx = int(fname.split('_pidx_')[-1].replace('.npy', ''))
            if p_idx < 0 or p_idx >= WIN_PTS:
                continue
            raw_data = np.load(fpath).astype(np.float32)
            if raw_data.shape[0] < WIN_PTS:
                pad = np.zeros((WIN_PTS - raw_data.shape[0], 3), dtype=np.float32)
                raw_data = np.vstack([raw_data, pad])
            else:
                raw_data = raw_data[:WIN_PTS, :3]

            X.append(normalize_clip(raw_data))
            Yc.append([0, 1])  # [Noise, EQ]
            Yp.append(generate_gaussian_label(WIN_PTS, p_idx))
        except Exception:
            continue
    return X, Yc, Yp


def sample_neg_from_arrays(arrays_list, target_count):
    X, Yc, Yp = [], [], []
    valid_arrays = [arr for arr in arrays_list if arr is not None and len(arr) >= WIN_PTS]
    if not valid_arrays or target_count <= 0:
        return X, Yc, Yp

    samples_per_arr = max(1, int(np.ceil(target_count / len(valid_arrays))))
    for data in valid_arrays:
        max_start = len(data) - WIN_PTS
        for _ in range(samples_per_arr):
            if len(X) >= target_count:
                return X, Yc, Yp
            start = random.randint(0, max_start) if max_start > 0 else 0
            clip = data[start:start + WIN_PTS]
            X.append(normalize_clip(clip))
            Yc.append([1, 0])  # [Noise, EQ]
            Yp.append(np.zeros((WIN_PTS, 1), dtype=np.float32))
    return X, Yc, Yp


def prepare_finetune_data():
    # 1. Event-level split for positive samples (.npy)
    all_pos_files = [
        f for f in glob.glob(os.path.join(POS_DATA_DIR, "**", "*_pidx_*.npy"), recursive=True)
        if "_pidx_-1" not in f and "PolicyC" not in f
    ]

    event_dict = {}
    for f in all_pos_files:
        evt_id = extract_event_id(os.path.basename(f))
        event_dict.setdefault(evt_id, []).append(f)

    unique_events = sorted(event_dict.keys())
    print(f"Positive files: {len(all_pos_files)} | Unique events: {len(unique_events)}")

    if len(unique_events) >= 2:
        train_events, val_events = train_test_split(
            unique_events, test_size=0.2, random_state=SEED
        )
        pos_train_files = [f for evt in train_events for f in event_dict[evt]]
        pos_val_files = [f for evt in val_events for f in event_dict[evt]]
    else:
        # Fallback for minimal demo datasets containing only 1 unique event
        pos_train_files, pos_val_files = train_test_split(
            all_pos_files, test_size=0.2, random_state=SEED
        )

    X_pos_tr, Yc_pos_tr, Yp_pos_tr = load_pos_from_files(pos_train_files)
    X_pos_va, Yc_pos_va, Yp_pos_va = load_pos_from_files(pos_val_files)
    print(f"Positive windows -> Train: {len(X_pos_tr)}, Val: {len(X_pos_va)}")

    # 2. Recording-level split for negative samples (.txt)
    neg_txt_files = sorted(
        glob.glob(os.path.join(NEG_HUMAN_DIR, "*.txt")) +
        glob.glob(os.path.join(NEG_SHAKE_DIR, "*.txt"))
    )

    if len(neg_txt_files) >= 2:
        neg_tr_files, neg_va_files = train_test_split(
            neg_txt_files, test_size=0.2, random_state=SEED
        )
        neg_tr_arrays = [clean_and_load_txt(f) for f in neg_tr_files]
        neg_va_arrays = [clean_and_load_txt(f) for f in neg_va_files]
    elif len(neg_txt_files) == 1:
        # Single-recording fallback: chronological split with a 10 s buffer
        arr = clean_and_load_txt(neg_txt_files[0])
        if arr is not None and len(arr) > 2 * WIN_PTS + BUFFER_PTS:
            split_idx = int(len(arr) * 0.8)
            neg_tr_arrays = [arr[:split_idx]]
            neg_va_arrays = [arr[split_idx + BUFFER_PTS:]]
        else:
            neg_tr_arrays, neg_va_arrays = [], []
    else:
        neg_tr_arrays, neg_va_arrays = [], []

    X_neg_tr, Yc_neg_tr, Yp_neg_tr = sample_neg_from_arrays(neg_tr_arrays, len(X_pos_tr))
    X_neg_va, Yc_neg_va, Yp_neg_va = sample_neg_from_arrays(neg_va_arrays, len(X_pos_va))
    print(f"Negative windows -> Train: {len(X_neg_tr)}, Val: {len(X_neg_va)}")

    # 3. Independently combine and shuffle train / val sets
    def finalize(Xp, Ycp, Ypp, Xn, Ycn, Ypn):
        combined = list(zip(Xp + Xn, Ycp + Ycn, Ypp + Ypn))
        if not combined:
            return np.array([]), np.array([]), np.array([])
        random.shuffle(combined)
        X, Yc, Yp = zip(*combined)
        return np.array(X, dtype=np.float32), np.array(Yc, dtype=np.int32), np.array(Yp, dtype=np.float32)

    X_tr, Yc_tr, Yp_tr = finalize(X_pos_tr, Yc_pos_tr, Yp_pos_tr, X_neg_tr, Yc_neg_tr, Yp_neg_tr)
    X_va, Yc_va, Yp_va = finalize(X_pos_va, Yc_pos_va, Yp_pos_va, X_neg_va, Yc_neg_va, Yp_neg_va)
    return X_tr, X_va, Yc_tr, Yc_va, Yp_tr, Yp_va


# ================= Fine-Tuning =================
def finetune_model(X_tr, X_va, Yc_tr, Yc_va, Yp_tr, Yp_va):
    print(f"\nLoading pre-trained model: {ORIGINAL_MODEL_PATH}")
    if not os.path.exists(ORIGINAL_MODEL_PATH):
        print("Error: Pre-trained model not found. Please run train.py first.")
        return

    model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH, compile=False)

    # Freeze bottom 60% layers to preserve general seismic feature extraction
    total_layers = len(model.layers)
    freeze_until = int(total_layers * 0.6)
    print(f"Model has {total_layers} layers; freezing the bottom {freeze_until} layers...")
    for i, layer in enumerate(model.layers):
        layer.trainable = (i >= freeze_until)

    out_names = [out.name.split('/')[0] for out in model.outputs]

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss={out_names[0]: focal_loss(), out_names[1]: 'binary_crossentropy'},
        loss_weights={out_names[0]: 0.5, out_names[1]: 1.0},
        metrics={out_names[0]: 'accuracy'}
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        verbose=1
    )

    os.makedirs(os.path.dirname(SAVE_FINETUNED_PATH), exist_ok=True)

    model.fit(
        x=X_tr,
        y={out_names[0]: Yc_tr, out_names[1]: Yp_tr},
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_va, {out_names[0]: Yc_va, out_names[1]: Yp_va}),
        callbacks=[early_stop],
        verbose=1
    )

    model.save(SAVE_FINETUNED_PATH)
    print(f"Fine-tuned model saved to: {SAVE_FINETUNED_PATH}")


if __name__ == "__main__":
    X_tr, X_va, Yc_tr, Yc_va, Yp_tr, Yp_va = prepare_finetune_data()
    if len(X_tr) > 0 and len(X_va) > 0:
        finetune_model(X_tr, X_va, Yc_tr, Yc_va, Yp_tr, Yp_va)
    else:
        print("Error: Insufficient samples to run fine-tuning.")