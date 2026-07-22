import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D, MaxPooling1D, 
                                     UpSampling1D, Concatenate, GlobalAveragePooling1D, 
                                     BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold
from tensorflow.keras.layers import SeparableConv1D, Activation


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.keras.mixed_precision.set_global_policy('float32')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_samples", "4_train_5s")
SAVE_DIR = os.path.join(BASE_DIR, "train_model")

# data parameter
SAMPLE_RATE = 100
WIN_LEN_SEC = 5
INPUT_LEN = int(SAMPLE_RATE * WIN_LEN_SEC)  
N_CHANNELS = 3
SIGMA = 10

# train parameter
BATCH_SIZE = 64
EPOCHS = 100
N_FOLDS = 5

def load_one_file(fpath_in):
    fpath = fpath_in.decode('utf-8')
    try:
        data = np.load(fpath)
    except:
        data = np.zeros((INPUT_LEN, N_CHANNELS), dtype=np.float32)
    
    max_val = np.max(np.abs(data))
    if max_val > 1e-6:
        data = data / max_val
    
    if data.shape[0] > INPUT_LEN:
        data = data[:INPUT_LEN, :]
    elif data.shape[0] < INPUT_LEN:
        pad = np.zeros((INPUT_LEN - data.shape[0], N_CHANNELS), dtype=np.float32)
        data = np.vstack([data, pad])
    
    data = data.astype(np.float32)
    
    fname = os.path.basename(fpath)
    is_negative = False
    pidx = -1
    
    if "_pidx_-1" in fname or "PolicyC" in fname:
        is_negative = True
    else:
        try:
            pidx = int(fname.split('_pidx_')[-1].split('.npy')[0])
        except:
            is_negative = True
    
    if pidx == -1:
        is_negative = True
    
    if is_negative:
        label_cls = np.array([1, 0], dtype=np.int32)
        label_pick = np.zeros((INPUT_LEN, 1), dtype=np.float32)
    else:
        label_cls = np.array([0, 1], dtype=np.int32)
        x_grid = np.arange(INPUT_LEN)
        gaussian = np.exp(-((x_grid - pidx)**2) / (2 * SIGMA**2))
        label_pick = gaussian.reshape(-1, 1).astype(np.float32)
    
    return data, label_cls, label_pick

@tf.function
def tf_augment(x, y_cls, y_pick):
    is_positive = tf.cast(y_cls[1], tf.bool)
    
    def augment_fn(img):
        scale = tf.random.uniform([], 0.5, 2.5)
        img = img * scale
        ch_scale = tf.random.uniform([1, 3], 0.8, 1.2)
        img = img * ch_scale
        noise = tf.random.normal(tf.shape(img), stddev=1.0) * tf.random.uniform([], 0.01, 0.05)
        img = img + noise
        m = tf.reduce_max(tf.abs(img))
        return tf.math.divide_no_nan(img, m)
    
    x_aug = tf.cond(is_positive, lambda: augment_fn(x), lambda: x)
    return x_aug, y_cls, y_pick

def tf_load_wrapper(fpath):
    X, yc, yp = tf.numpy_function(
        load_one_file, [fpath], [tf.float32, tf.int32, tf.float32]
    )
    X.set_shape((INPUT_LEN, N_CHANNELS))
    yc.set_shape((2,))
    yp.set_shape((INPUT_LEN, 1))
    return X, yc, yp

def create_balanced_train_dataset(pos_files, neg_files, batch_size):
    """trainset：1:1 balance"""
    ds_pos = tf.data.Dataset.from_tensor_slices(pos_files)\
        .shuffle(len(pos_files), reshuffle_each_iteration=True)\
        .repeat()
    
    ds_neg = tf.data.Dataset.from_tensor_slices(neg_files)\
        .shuffle(len(neg_files), reshuffle_each_iteration=True)\
        .repeat()
    
    ds_pos = ds_pos.map(tf_load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds_neg = ds_neg.map(tf_load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = tf.data.Dataset.zip((ds_pos, ds_neg))\
        .flat_map(lambda p, n: tf.data.Dataset.from_tensors(p)
                                     .concatenate(tf.data.Dataset.from_tensors(n)))
    
    ds = ds.map(tf_augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.map(lambda x, yc, yp: (x, {"cls_out": yc, "pick_out": yp}),
                num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def create_natural_val_dataset(pos_files, neg_files, batch_size):
    """validset：imbalance"""
    all_files = np.concatenate([pos_files, neg_files])
    
    ds = tf.data.Dataset.from_tensor_slices(all_files)\
        .shuffle(len(all_files))
    
    ds = ds.map(tf_load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.map(lambda x, yc, yp: (x, {"cls_out": yc, "pick_out": yp}),
                num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_cnn_model(input_shape=(500, 3)):

    inputs = Input(shape=input_shape)
    
    def mobile_block(x, filters, kernel_size):
        x = SeparableConv1D(filters, kernel_size, padding='same', depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    x1 = mobile_block(inputs, 8, 7)  # 9 -> 7
    p1 = MaxPooling1D(2)(x1)         # 500 -> 250
    
    x2 = mobile_block(p1, 16, 5)     # 7 -> 5
    p2 = MaxPooling1D(2)(x2)         # 250 -> 125

    x3 = mobile_block(p2, 32, 3)     # 5 -> 3, 125 -> 125

    u2 = UpSampling1D(2)(x3)         # 125 -> 250
    c2 = Concatenate()([u2, x2])
    d2 = mobile_block(c2, 16, 5)     # 16 filters
    
    u1 = UpSampling1D(2)(d2)         # 250 -> 500
    c1 = Concatenate()([u1, x1])
    d1 = mobile_block(c1, 8, 7)      # 8 filters
    

    pick_out = Conv1D(1, 1, activation='sigmoid', name='pick_out')(d1)
    
    gap = GlobalAveragePooling1D()(x3)
    dense1 = Dense(16, activation='relu')(gap) # 
    dense1 = Dropout(0.2)(dense1)
    cls_out = Dense(2, activation='softmax', name='cls_out')(dense1)
    
    return Model(inputs, [cls_out, pick_out])

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        
        return tf.reduce_sum(weight * cross_entropy, axis=-1)
    return loss

def plot_history(history, fold, save_dir):
    try:
        metrics = ['loss', 'cls_out_accuracy', 'pick_out_loss']
        plt.figure(figsize=(15, 5))
        for i, metric in enumerate(metrics):
            if metric in history.history:
                plt.subplot(1, 3, i+1)
                plt.plot(history.history[metric], label='Train')
                plt.plot(history.history[f'val_{metric}'], label='Val')
                plt.title(f'Fold {fold} - {metric}')
                plt.xlabel('Epoch')
                plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"history_fold_{fold}.png"))
        plt.close()
    except Exception as e:
        print(f"Plotting Error: {e}")

def plot_prediction_sample(model, pos_files_list, fold, save_dir):
    try:
        fpath = np.random.choice(pos_files_list)
        data = np.load(fpath)
        data = data / (np.max(np.abs(data)) + 1e-6)
        
        if len(data) < 500:
            data = np.vstack([data, np.zeros((500-len(data), 3))])
        else:
            data = data[:500]
        
        fname = os.path.basename(fpath)
        try:
            pidx = int(fname.split('_pidx_')[-1].split('.npy')[0])
        except:
            return
        
        inp = data[np.newaxis, ...].astype(np.float32)
        preds = model.predict(inp, verbose=0)
        pred_pick = preds[1][0]
        
        plt.figure(figsize=(10, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(data[:, 2], 'k', label='Z Component')
        plt.axvline(pidx, color='g', linestyle='--', label=f'True P ({pidx})')
        plt.axvline(np.argmax(pred_pick), color='r', linestyle='--', 
                   label=f'Pred P ({np.argmax(pred_pick)})')
        plt.title(f"Fold {fold} Sample Prediction")
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(pred_pick, 'r', label='Pred Probability')
        x_grid = np.arange(500)
        gauss = np.exp(-((x_grid - pidx)**2) / (2 * 10**2))
        plt.plot(gauss, 'g', alpha=0.5, label='True Gaussian')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"pred_sample_fold_{fold}.png"))
        plt.close()
    except Exception as e:
        print(f"Sample Plotting Error: {e}")

def evaluate_mae_stream(model, pos_files):
    ds = tf.data.Dataset.from_tensor_slices(pos_files)
    ds = ds.map(tf_load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, yc, yp: (x, yp)).batch(64)
    
    mae_list = []
    for x_batch, yp_batch in ds:
        preds = model.predict(x_batch, verbose=0)
        p_pred = preds[1]
        
        t_idx = np.argmax(yp_batch.numpy(), axis=1).flatten()
        p_idx = np.argmax(p_pred, axis=1).flatten()
        mae_list.extend(np.abs(t_idx - p_idx))
    
    return np.mean(mae_list)

def run_pipeline():
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.npy"), recursive=True)
    
    pos_files = np.array([f for f in all_files 
                          if "_pidx_-1" not in f and "PolicyC" not in f])
    neg_files = np.array([f for f in all_files if f not in pos_files])
    
    print(f"Pos Samples: {len(pos_files)}")
    print(f"Neg Samples: {len(neg_files)}")
    
    if len(pos_files) == 0:
        print("Error: No positive samples found.")
        return
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    results = []
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(pos_files)):
        print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")
        
        train_pos = pos_files[tr_idx]
        val_pos = pos_files[val_idx]
        
        np.random.shuffle(neg_files)

        desired_val_neg = len(val_pos) * 16
        
        if desired_val_neg >= len(neg_files):
            val_neg_count = max(1, len(neg_files) // 5)
        else:
            val_neg_count = desired_val_neg
            
        val_neg = neg_files[:val_neg_count]
        train_neg = neg_files[val_neg_count:]
        
        train_ds = create_balanced_train_dataset(train_pos, train_neg, BATCH_SIZE)
        val_ds = create_natural_val_dataset(val_pos, val_neg, BATCH_SIZE)
        
        steps = int(np.ceil(len(train_pos) / (BATCH_SIZE / 2)))
        val_steps = int(np.ceil((len(val_pos) + len(val_neg)) / BATCH_SIZE))
        
        print(f"Steps: {steps}, Val Steps: {val_steps}")
        
        model = build_cnn_model()
        model.compile(
            optimizer="adam",
            loss={
                "cls_out": focal_loss(),
                "pick_out": "binary_crossentropy"
            },
            loss_weights={"cls_out": 0.5, "pick_out": 1.0},
            metrics={"cls_out": ["accuracy"]}
        )
        
        callbacks = [
            EarlyStopping(
                monitor="val_pick_out_loss",
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor="val_pick_out_loss",
                factor=0.5,
                patience=5
            ),
            ModelCheckpoint(
                f"{SAVE_DIR}/fold_{fold}.h5",
                save_best_only=True,
                monitor="val_pick_out_loss"
            )
        ]
        
        history = model.fit(
            train_ds,
            steps_per_epoch=steps,
            validation_data=val_ds,
            validation_steps=val_steps,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        
        # save and plot
        pd.DataFrame(history.history).to_csv(
            f"{SAVE_DIR}/history_fold_{fold}.csv", index=False
        )
        plot_history(history, fold, SAVE_DIR)
        plot_prediction_sample(model, val_pos, fold, SAVE_DIR)
        
        # evaluate
        print("Calculating MAE...")
        mae = evaluate_mae_stream(model, val_pos)
        print(f"Fold {fold} Positive MAE: {mae:.2f} pts")
        results.append({"fold": fold, "mae": mae})
        
        tf.keras.backend.clear_session()
        del model
    
    pd.DataFrame(results).to_csv(f"{SAVE_DIR}/final_results.csv", index=False)
    print("\nDone!")

if __name__ == "__main__":
    run_pipeline()