# prediction.py
import os
import io
import re
import gc
import sqlite3
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, lfilter, lfilter_zi
from collections import deque

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Options: 'human_noise' | 'shaking_table' | 'continuous_eq'
DATA_MODE = 'human_noise'  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "fold_0_v4.h5")
DB_PATH = os.path.join(BASE_DIR, "data_input", "earthquake.db")
INPUT_ROOT = os.path.join(BASE_DIR, "data_input", "phone_records",DATA_MODE)  
OUTPUT_DIR = os.path.join(BASE_DIR, "data_output", DATA_MODE)

ENABLE_PLOTTING = True   #画图开关，默认关闭

FS = 100.0
STA_LEN = 1.0
LTA_LEN = 10.0
THRESHOLD_ON = 3.5
THRESHOLD_OFF = 2.0
COOLDOWN_SEC = 3.0
SHUTDOWN_WAIT_SEC = 5.0

CNN_WIN_LEN = 5.0
CNN_INPUT_PTS = int(CNN_WIN_LEN * FS)  
SEARCH_PRE_SEC = 6.0    
SEARCH_POST_SEC = 2.0   
SLIDE_STEP_SEC = 0.5    

EQ_PROB_THRESHOLD = 0.58  


def extract_hour_from_filename(filename, trigger_time_s):
    match = re.search(r'\d{8}(\d{2})\.txt', filename)
    if match:
        base_hour = int(match.group(1))
        return (base_hour + trigger_time_s / 3600.0) % 24.0
    return None

def clean_and_load_txt(fpath):
    if DATA_MODE == 'shaking_table':
        return pd.read_csv(fpath, sep=r'\s+', header=None, engine='c')
        
    clean_lines = []
    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            lower_line = line.strip().lower()
            if lower_line.startswith('location') or lower_line.startswith('unknown') or lower_line == '':
                continue
            clean_lines.append(line)
            
    if len(clean_lines) < FS * 3: 
        return None
    try:
        return pd.read_csv(io.StringIO(''.join(clean_lines)), sep=r'\s+', header=None, engine='python', on_bad_lines='skip')
    except Exception:
        return pd.read_csv(io.StringIO(''.join(clean_lines)), sep=r'\s+', header=None, engine='python', on_bad_lines='skip')


class OptimizedStreamDetector:
    def __init__(self):
        print(f"[INFO] Loading model: {MODEL_PATH}")
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        nyq = 0.5 * FS
        self.b, self.a = butter(4, [1/nyq, 20/nyq], btype='band')
        self.nsta = int(STA_LEN * FS)
        self.nlta = int(LTA_LEN * FS)
        self.raw_buffer_maxlen = int((SEARCH_PRE_SEC + CNN_WIN_LEN + 5.0) * FS)
        self.reset()

    def reset(self):
        self.zi = [lfilter_zi(self.b, self.a) * 0.0 for _ in range(3)]
        self.eng_buffer = np.zeros(self.nlta, dtype=np.float32)
        self.buf_ptr = 0
        self.sum_sta = 0.0
        self.sum_lta = 0.0
        self.raw_data_buffer = deque(maxlen=self.raw_buffer_maxlen)
        self.frozen_lta = 0.0
        self.triggered = False
        self.cooldown_counter = 0
        self.cnn_countdown = 0
        self.current_event_start = 0
        self.shutdown_counter = 0

    def warm_up(self, noise_level=1.0, duration_sec=15.0):
        np.random.seed(42)
        n = int(duration_sec * FS)
        noise = np.random.normal(0, noise_level, (n, 3)).astype(np.float32)
        filt_data = np.zeros_like(noise)
        for ch in range(3):
            filt_data[:, ch], self.zi[ch] = lfilter(self.b, self.a, noise[:, ch], zi=self.zi[ch])
        filt_ud = filt_data[:, 2]
        for val in filt_ud: 
            self._update_sma_sums(val ** 2)
        for val in noise: 
            self.raw_data_buffer.append(val)

    def _update_sma_sums(self, new_eng):
        oldest_lta = self.eng_buffer[self.buf_ptr]
        sta_pop_idx = (self.buf_ptr - self.nsta) % self.nlta
        oldest_sta = self.eng_buffer[sta_pop_idx]
        self.sum_lta = self.sum_lta - oldest_lta + new_eng
        self.sum_sta = self.sum_sta - oldest_sta + new_eng
        self.eng_buffer[self.buf_ptr] = new_eng
        self.buf_ptr = (self.buf_ptr + 1) % self.nlta

    def process_data_stream(self, file_data, filename):
        n = len(file_data)
        ratio_history = np.zeros(n, dtype=np.float32)
        filt_ud_history = np.zeros(n, dtype=np.float32)
        events = []
        
        filt_data = np.zeros_like(file_data, dtype=np.float32)
        for ch in range(3):
            filt_data[:, ch], self.zi[ch] = lfilter(self.b, self.a, file_data[:, ch], zi=self.zi[ch])
        ud_filt = filt_data[:, 2]
        
        for i in range(n):
            val_raw = file_data[i]
            val_filt = ud_filt[i]
            filt_ud_history[i] = val_filt
            self.raw_data_buffer.append(val_raw)
            self._update_sma_sums(val_filt ** 2)
            
            sta_val = self.sum_sta / self.nsta
            curr_lta = self.frozen_lta if self.triggered else (self.sum_lta / self.nlta)
            if curr_lta < 1e-12: curr_lta = 1e-12
            
            ratio = sta_val / curr_lta
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                ratio_history[i] = 0.0
                continue
            ratio_history[i] = ratio
            
            if not self.triggered:
                if ratio > THRESHOLD_ON:
                    self.triggered = True
                    self.frozen_lta = curr_lta
                    self.current_event_start = max(0, i - int(0.5 * FS))
                    self.cnn_countdown = int(SEARCH_POST_SEC * FS)
                    self.shutdown_counter = 0
            else:
                if self.cnn_countdown > 0:
                    self.cnn_countdown -= 1
                    if self.cnn_countdown == 0:
                        trigger_idx = i - int(SEARCH_POST_SEC * FS)
                        self._scan_and_pick_cnn(i, trigger_idx, events, filename, ratio_history)
                
                if ratio < THRESHOLD_OFF:
                    self.shutdown_counter += 1
                    if self.shutdown_counter > int(SHUTDOWN_WAIT_SEC * FS):
                        self.triggered = False
                        self.cooldown_counter = int(COOLDOWN_SEC * FS)
                        self.shutdown_counter = 0
                        if events: events[-1]['event_end_idx'] = i
                else:
                    self.shutdown_counter = 0
        return ratio_history, filt_ud_history, events

    def _scan_and_pick_cnn(self, current_stream_idx, trigger_idx, events_list, filename, ratio_history):
        buffer_arr = np.array(self.raw_data_buffer, dtype=np.float32)
        buf_end = len(buffer_arr)
        trigger_buf_idx = buf_end - int(SEARCH_POST_SEC * FS)
        
        # Mid-End: Physical Gating Filters
        check_len = int(1.0 * FS)  
        if trigger_buf_idx >= 0:
            end_check = min(trigger_buf_idx + check_len, buf_end)
            check_win = buffer_arr[trigger_buf_idx : end_check]
            
            if len(check_win) > 0:
                check_win_demean = check_win - np.mean(check_win, axis=0)
                x_data, y_data, z_data = check_win_demean[:, 0], check_win_demean[:, 1], check_win_demean[:, 2]
                
                max_x, max_y, max_z = np.max(np.abs(x_data)), np.max(np.abs(y_data)), np.max(np.abs(z_data))
                max_vector = np.max(np.sqrt(x_data**2 + y_data**2 + z_data**2))
                sustained_count = np.sum(ratio_history[trigger_idx : current_stream_idx] > 2.0)

                rms_x = np.sqrt(np.mean(x_data**2))
                rms_y = np.sqrt(np.mean(y_data**2))
                rms_z = np.sqrt(np.mean(z_data**2))
                
                total_rms = rms_x + rms_y + rms_z + 1e-10
                z_ratio = rms_z / total_rms
                
                reject_reason = None
                
                if max_vector < 0.003:
                    reject_reason = "2a_amplitude_too_low"
                    
                elif sustained_count < int(0.3 * FS):
                    reject_reason = "2b_duration_too_short"
                    
                elif z_ratio > 0.85:
                    reject_reason = "2c_vertical_dominated"
                    
                elif rms_x / (rms_y + 1e-10) > 5.0 or rms_y / (rms_x + 1e-10) > 5.0:
                    reject_reason = "2d_horizontal_asymmetric"
                
                if reject_reason:
                    events_list.append({
                        'file': filename, 'event_type': 'noise', 'reject_reason': reject_reason, 
                        'trigger_time_s': trigger_idx / FS, 'trigger_time_idx': trigger_idx,
                        'p_time_s': -1, 'p_prob': 0.0, 'trigger_to_p_delay_s': -999,
                        'event_start_idx': self.current_event_start, 'event_end_idx': current_stream_idx + int(2.0*FS)
                    })
                    return  
        
        search_end_idx = buf_end
        search_start_idx = buf_end - int((SEARCH_POST_SEC + SEARCH_PRE_SEC) * FS)
        if search_start_idx < 0: return
            
        win_pts, step_pts = CNN_INPUT_PTS, int(SLIDE_STEP_SEC * FS)
        batch_clips, batch_start_indices = [], []
        
        curr = search_start_idx
        while curr + win_pts <= search_end_idx:
            clip = buffer_arr[curr : curr + win_pts]
            clip_demean = clip - np.mean(clip, axis=0)
            max_val = np.max(np.abs(clip_demean))
            clip_norm = clip_demean / max_val if max_val > 0.02 else clip_demean / 0.02
            
            batch_clips.append(clip_norm)
            batch_start_indices.append(curr)
            curr += step_pts
        
        if not batch_clips: return
        batch_clips = np.array(batch_clips, dtype=np.float32)
        preds = self.model.predict(batch_clips, verbose=0)
        probs_eq, pick_curves = preds[0][:, 1], preds[1]
        
        best_candidate, best_score = None, -1.0
        for i in range(len(batch_clips)):
            prob = float(probs_eq[i])
            if prob < 0.3: continue 
                
            p_rel_idx = np.argmax(pick_curves[i])
            dt = (batch_start_indices[i] + p_rel_idx - trigger_buf_idx) / FS
            
            time_weight = 1.0 if -5.0 <= dt <= 0.2 else (0.1 if dt > 0.2 else 0.5)
            score = prob * time_weight
            if score > best_score:
                best_score = score
                best_candidate = {'prob': prob, 'p_buf_idx': batch_start_indices[i] + p_rel_idx, 'dt': dt}
        
        if best_candidate and best_candidate['prob'] >= EQ_PROB_THRESHOLD and best_score > 0.2:
            p_stream_idx = current_stream_idx + (best_candidate['p_buf_idx'] - buf_end)
            events_list.append({
                'file': filename, 'event_type': 'earthquake', 'reject_reason': 'None',
                'trigger_time_s': trigger_idx / FS, 'trigger_time_idx': trigger_idx,
                'p_time_s': p_stream_idx / FS, 'p_prob': best_candidate['prob'],
                'trigger_to_p_delay_s': best_candidate['dt'], 'event_start_idx': self.current_event_start,
                'event_end_idx': current_stream_idx + int(2.0*FS)
            })
        else:
            events_list.append({
                'file': filename, 'event_type': 'noise', 'reject_reason': '3_low_cnn_score',
                'trigger_time_s': trigger_idx / FS, 'trigger_time_idx': trigger_idx,
                'p_time_s': -1, 'p_prob': best_candidate['prob'] if best_candidate else 0.0,
                'trigger_to_p_delay_s': best_candidate['dt'] if best_candidate else -999,
                'event_start_idx': self.current_event_start, 'event_end_idx': current_stream_idx + int(2.0*FS)
            })


def plot_stream_result(filename, raw_ud, filt_ud, ratio, events, save_dir, true_p_idx=None, title_prefix=""):
    plt.close('all')
    times = np.arange(len(raw_ud)) / FS
    eq_events = [e for e in events if e['event_type'] == 'earthquake']
    n_eq = len(eq_events)
    
    fig = plt.figure(figsize=(14, 10 + n_eq * 4))
    gs = gridspec.GridSpec(2 + n_eq, 1, figure=fig, hspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(f"{title_prefix}: {filename} | EQ Found: {n_eq}", fontsize=20, fontweight='bold', pad=25)
    ax1.plot(times, raw_ud, 'lightgray', lw=0.8, label='Raw UD')
    ax1.plot(times, filt_ud, 'k', lw=0.8, label='Filtered UD')
    ax1.set_ylabel("Acc ($m/s^2$)", fontsize=18) 
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(times[0], times[-1]) 
    ax1.tick_params(axis='both', labelsize=14)
    
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(times, ratio, 'g', lw=1, label='STA/LTA Ratio')
    ax2.axhline(THRESHOLD_ON, color='r', ls='--', label='On Threshold')
    ax2.axhline(THRESHOLD_OFF, color='orange', ls=':', label='Off Threshold')
    ax2.set_ylabel("Ratio", fontsize=18)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=14)
    
    if true_p_idx is not None and 0 <= true_p_idx < len(raw_ud):
        ax1.axvline(true_p_idx / FS, color='green', lw=2.0, alpha=0.8, label='True P')

    for evt in events:
        trig = evt['trigger_time_s']
        is_eq = (evt['event_type'] == 'earthquake')
        c, a = ('r', 0.15) if is_eq else ('gray', 0.05)
        
        if 'event_start_idx' in evt:
            ax1.axvspan(evt['event_start_idx']/FS, evt['event_end_idx']/FS, color=c, alpha=a)
            ax2.axvspan(evt['event_start_idx']/FS, evt['event_end_idx']/FS, color=c, alpha=a)
        
        ax1.axvline(trig, color='blue', ls=':', alpha=0.5, label='STA/LTA Trigger')
        ax2.axvline(trig, color='blue', ls=':', alpha=0.5, label='STA/LTA Trigger')
    
        if is_eq and evt['p_time_s'] > 0:
            p_t = evt['p_time_s']
            ax1.axvline(p_t, color='red', lw=1.5, ls='--', label='Pred P')
            y_max = np.percentile(np.abs(filt_ud), 99.5) if len(filt_ud) > 0 else 1
            ax1.text(p_t-2, y_max, f"P at {p_t:.2f}s\nProb={evt['p_prob']:.2f}", color='darkred', fontsize=14, fontweight='bold', ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", alpha=0.8))

    for i, evt in enumerate(eq_events):
        p_t = evt['p_time_s']
        ax_zoom = fig.add_subplot(gs[2+i, 0])
        idx_s, idx_e = int(max(0, p_t - 3.0) * FS), int(min(times[-1], p_t + 3.0) * FS)
        ax_zoom.plot(times[idx_s:idx_e], filt_ud[idx_s:idx_e], 'k', lw=1.2, label='Filtered')
        ax_zoom.axvline(p_t, color='red', lw=2.0, ls='--', label='Pred P')
        ax_zoom.axvline(evt['trigger_time_s'], color='blue', lw=1.5, ls=':', label='STA/LTA Trigger')
        if true_p_idx is not None and max(0, p_t - 3.0) <= true_p_idx / FS <= min(times[-1], p_t + 3.0):
            ax_zoom.axvline(true_p_idx / FS, color='green', lw=2.5, alpha=0.6, label='True P')
        ax_zoom.set_title(f"Event #{i+1} Zoom (P at {p_t:.2f}s | Prob={evt['p_prob']:.2f})", fontsize=18, color='darkred', fontweight='bold')
        ax_zoom.set_ylabel("Acc", fontsize=18)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.tick_params(axis='both', labelsize=14)

    plt.xlabel("Time (s)", fontsize=18)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
    plt.close('all')

def plot_p_error_distribution(tp_df, save_dir):
    if len(tp_df) == 0: return
    errors_sec = (tp_df["pre_pidx"] - tp_df["idx"]) / FS
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors_sec, bins=50, range=(-2, 2), color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(errors_sec.mean(), color="r", ls="-", lw=2, label=f"Mean={errors_sec.mean():.3f}s")
    ax.set_xlabel("Error (s) = Pred - True", fontsize=18)
    ax.set_ylabel("Count", fontsize=18)
    ax.set_title(f"P-Wave Picking Error Distribution (N={len(tp_df)})", fontsize=20, fontweight='bold')
    ax.legend(fontsize=16)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "p_pick_error_dist.png"), dpi=300)
    plt.close()


def run_file_based_pipeline(detector):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file_path = os.path.join(OUTPUT_DIR,f"processed_{DATA_MODE}_files.log")
    processed_files = set()
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r', encoding='utf-8') as f:
            processed_files = set(line.strip() for line in f)

    terminals = sorted([p for p in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, p)) and not p.startswith('.')])
    summary_stats = []

    for term in terminals:
        term_dir, save_dir = os.path.join(INPUT_ROOT, term), os.path.join(OUTPUT_DIR, term)
        files = sorted([f for f in os.listdir(term_dir) if f.endswith(".txt") and not f.startswith('.')])
        if not files: continue
            
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "res_events.csv")
        for f in files:
            fpath = os.path.join(term_dir, f)
            if fpath in processed_files: continue
            try:
                df = clean_and_load_txt(fpath)
                if df is None or df.shape[1] < 3: continue
                raw = df.iloc[:, -3:].values.astype(np.float32)

                detector.reset()
                noise_lvl = max(1e-10, np.median(np.abs(raw[:int(3.0*FS), 2])))
                detector.warm_up(noise_level=noise_lvl, duration_sec=15.0)
                ratio, filt, events = detector.process_data_stream(raw, f)
                
                if events:
                    if ENABLE_PLOTTING:
                        plot_stream_result(f, raw[:, 2], filt, ratio, events, save_dir, title_prefix=DATA_MODE.upper())
                    pd.DataFrame(events).to_csv(csv_path, mode='a' if os.path.exists(csv_path) else 'w', header=not os.path.exists(csv_path), index=False)
                with open(log_file_path, 'a', encoding='utf-8') as log_f: log_f.write(f"{fpath}\n")
            except Exception as e:
                print(f"[Error] Failed on file {f}: {e}")
                
        if os.path.exists(csv_path):
            tdf = pd.read_csv(csv_path, on_bad_lines='skip')
            summary_stats.append({
                'Terminal_Model': term, 'Total_STA_LTA_Triggers': len(tdf),
                'CNN_Confirmed_EQs': len(tdf[tdf['event_type'] == 'earthquake']),
                'Rejection_Rate(%)': round((len(tdf) - len(tdf[tdf['event_type'] == 'earthquake'])) / max(1, len(tdf)) * 100, 2)
            })

    if summary_stats:
        pd.DataFrame(summary_stats).to_csv(os.path.join(OUTPUT_DIR, f"summary_statistics.csv"), index=False, encoding='utf-8-sig')

def run_database_pipeline(detector):
    MISSED_PLOT_DIR = os.path.join(OUTPUT_DIR, "Missed_EQ_Diagnostic_Plots")
    os.makedirs(MISSED_PLOT_DIR, exist_ok=True)
    CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "seismic_processing_checkpoint.csv")
    
    if os.path.exists(CHECKPOINT_FILE):
        df = pd.read_csv(CHECKPOINT_FILE)
    else:
        if not os.path.exists(DB_PATH): return
        conn = sqlite3.connect(DB_PATH)
        df_raw = pd.read_sql("SELECT * FROM origi_raw_mapping", conn)
        conn.close()
        df = df_raw.dropna(subset=["currenten_fpath"]).copy()
        df["pre_class"], df["pre_pidx"], df["pre_prob"], df["status"] = pd.NA, pd.NA, pd.NA, pd.NA

    missed_analysis_list, newly_processed = [], 0
    for i, row in df.iterrows():
        if pd.notna(row.get("status")): continue
        raw_path = row["currenten_fpath"]

        if not os.path.exists(raw_path): continue
        try:
            raw_data = np.loadtxt(raw_path, usecols=(0, 1, 2))
            if raw_data.ndim == 1 or len(raw_data) < int(FS * 5.5): continue
                
            detector.reset()
            detector.warm_up(noise_level=max(1e-10, np.median(np.abs(raw_data[:int(3.0*FS), 2]))), duration_sec=15.0)
            
            filename = os.path.basename(raw_path)
            ratio, filt, events = detector.process_data_stream(raw_data, filename)
            valid_cands = [e for e in events if e['event_type'] == 'earthquake' and e['p_time_s'] >= 0]
            
            pred_class = 1 if len(valid_cands) > 0 else 0
            final_prob, final_pidx = 0.0, -1
            
            if pred_class == 1:
                target_idx = row["idx"]
                matched_evt = valid_cands[np.argmin([abs(int(round(e['p_time_s'] * FS)) - target_idx) for e in valid_cands])]
                final_prob = matched_evt['p_prob']
                final_pidx = int(round(matched_evt['p_time_s'] * FS))
                
            df.at[i, "pre_class"], df.at[i, "pre_prob"], df.at[i, "pre_pidx"], df.at[i, "status"] = pred_class, final_prob, final_pidx, "processed"
            newly_processed += 1

            if pred_class == 0:
                reason = "1_STA_LTA_trigger_failed" if len(events) == 0 else ("3_low_cnn_score" if any(e.get('trigger_to_p_delay_s') != -999 for e in events) else next((e.get('reject_reason') for e in events if 'reject_reason' in e), "2_gate_blocked"))
                missed_analysis_list.append({'Year': row.get('year', 'Unknown'), 'Filename': filename, 'Interception_Stage': reason})
                
                if ENABLE_PLOTTING:
                    plot_stream_result(filename, raw_data[:, 2], filt, ratio, events, MISSED_PLOT_DIR, true_p_idx=row["idx"], title_prefix="MISSED EQ")
            else:
                if ENABLE_PLOTTING:
                    HIT_PLOT_DIR = os.path.join(OUTPUT_DIR, "Hit_EQ_Plots", str(row["year"]))
                    os.makedirs(HIT_PLOT_DIR, exist_ok=True)
                    plot_stream_result(filename, raw_data[:, 2], filt, ratio, events, HIT_PLOT_DIR, true_p_idx=row["idx"], title_prefix="HIT EQ")
                    
            if newly_processed % 50 == 0: df.to_csv(CHECKPOINT_FILE, index=False)
        except Exception:
            df.at[i, "status"] = "error"

    df.to_csv(CHECKPOINT_FILE, index=False)
    vdf = df[df["status"] == "processed"].copy()
    if len(vdf) > 0:
        tp_df = vdf[vdf["pre_class"] == 1].copy()
        plot_p_error_distribution(tp_df, OUTPUT_DIR)
    if missed_analysis_list:
        pd.DataFrame(missed_analysis_list).to_csv(os.path.join(OUTPUT_DIR, "Missed_EQ_Diagnostic_Details.csv"), index=False, encoding='utf-8-sig')

def main():
    print(f"[INFO] Operational Mode Selected: [{DATA_MODE.upper()}]")
    detector = OptimizedStreamDetector()
    if DATA_MODE in ['human_noise', 'shaking_table']:
        run_file_based_pipeline(detector)
    elif DATA_MODE == 'continuous_eq':
        run_database_pipeline(detector)


if __name__ == "__main__":
    main()