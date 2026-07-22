import os
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


POS_DATA_DIR = os.path.join(BASE_DIR, "data_samples", "4_train_5s") 
NEG_HUMAN_DIR = os.path.join(BASE_DIR, "data_samples", "3_human_daily") 
NEG_SHAKE_DIR = os.path.join(BASE_DIR, "data_samples", "2_shaking_table")

ORIGINAL_MODEL_PATH = os.path.join(BASE_DIR, "train_model", "fold_0.h5")  
SAVE_FINETUNED_PATH = os.path.join(BASE_DIR, "fine_tune_model", "finetuned_v1.h5")

FS = 100.0
WIN_PTS = 500  
LEARNING_RATE = 1e-4  
EPOCHS = 100
BATCH_SIZE = 32


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
    y = np.exp(-(x - p_idx)**2 / (2 * sigma**2))
    return y.reshape(-1, 1)

def load_txt_to_numpy(fpath):
    try:
        data = np.loadtxt(fpath)
        if data.ndim == 1: data = data.reshape(-1, 1)

        return data[:, -3:].astype(np.float32)
    except:
        return None

def normalize_clip(clip):
    clip_demean = clip - np.mean(clip, axis=0)
    max_v = np.max(np.abs(clip_demean))

    return clip_demean / max_v

def prepare_finetune_data():

    X_pos, Yc_pos, Yp_pos = [], [], []
    X_neg, Yc_neg, Yp_neg = [], [], []


    pos_files = glob.glob(os.path.join(POS_DATA_DIR, "**", "*_pidx_*.npy"), recursive=True)
    
    for fpath in pos_files:
        fname = os.path.basename(fpath)
        try:
            p_idx = int(fname.split('_pidx_')[-1].replace('.npy', ''))
            if p_idx < 0 or p_idx >= WIN_PTS: continue
            
            raw_data = np.load(fpath)
            norm_data = normalize_clip(raw_data)
            
            X_pos.append(norm_data)
            Yc_pos.append([0, 1]) # [Noise, EQ]
            Yp_pos.append(generate_gaussian_label(WIN_PTS, p_idx))
        except: continue

    pos_count = len(X_pos)


    neg_txt_files = glob.glob(os.path.join(NEG_HUMAN_DIR, "*.txt")) + \
                    glob.glob(os.path.join(NEG_SHAKE_DIR, "*.txt"))
    
    if len(neg_txt_files) > 0 and pos_count > 0:
        samples_per_file = max(1, pos_count // len(neg_txt_files))
        
        for fpath in neg_txt_files:
            data = load_txt_to_numpy(fpath)
            if data is None or len(data) <= WIN_PTS: continue
            
            for _ in range(samples_per_file):
                start = random.randint(0, len(data) - WIN_PTS)
                clip = data[start : start + WIN_PTS]
                norm_data = normalize_clip(clip)
                
                X_neg.append(norm_data)
                Yc_neg.append([1, 0]) # [Noise, EQ]
                Yp_neg.append(np.zeros((WIN_PTS, 1)))


    X_all = np.array(X_pos + X_neg)
    Yc_all = np.array(Yc_pos + Yc_neg)
    Yp_all = np.array(Yp_pos + Yp_neg)

    return train_test_split(X_all, Yc_all, Yp_all, test_size=0.2, random_state=SEED, shuffle=True)


def finetune_model(X_tr, X_va, Yc_tr, Yc_va, Yp_tr, Yp_va):
    print(f"\n load original model: {ORIGINAL_MODEL_PATH}")
    if not os.path.exists(ORIGINAL_MODEL_PATH):
        print("No model，run train.py first")
        return

    model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH, compile=False)

    total_layers = len(model.layers)
    freeze_until = int(total_layers * 0.6)
    
    print(f"Model has {total_layers} layers，freezing {freeze_until} ...")
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
        patience=10, 
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
    print("Done")

if __name__ == "__main__":
    X_tr, X_va, Yc_tr, Yc_va, Yp_tr, Yp_va = prepare_finetune_data()
    if len(X_tr) > 0:
        finetune_model(X_tr, X_va, Yc_tr, Yc_va, Yp_tr, Yp_va)