import os
import io
import re
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, lfilter, lfilter_zi
from collections import deque
import glob

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# Options: 'human_noise' | 'shaking_table' | 'continuous_eq'
DATA_MODE = 'human_noise'  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "paper_cnn_model", "fold_0_v4.h5")

INPUT_ROOT_MAP = {
    'continuous_eq': os.path.join(BASE_DIR, "data_samples", "1_continuous data"),
    'shaking_table': os.path.join(BASE_DIR, "data_samples", "2_shaking_table"),
    'human_noise': os.path.join(BASE_DIR, "data_samples", "3_human_daily")
}
INPUT_DIR = INPUT_ROOT_MAP[DATA_MODE]
OUTPUT_DIR = os.path.join(BASE_DIR, "pred_output", DATA_MODE)

ENABLE_PLOTTING = True   

FS = 100.0
STA_LEN = 1.0
LTA_LEN = 10.0
THRESHOLD_ON = 3.5
THRESHOLD_OFF = 2.0
COOLDOWN_SEC = 3.0
SHUTDOWN_WAIT_SEC = 5.0

CNN_WIN_LEN = 5.0
CNN_INPUT_PTS = int(CNN_WIN_LEN * FS)  
SEARCH_PRE_SEC = 4    
SEARCH_POST_SEC = 2.0   
SLIDE_STEP_SEC = 0.5    

EQ_PROB_THRESHOLD = 0.57  

def load_trace_data(fpath):
    if fpath.endswith('.npy'):
        data = np.load(fpath)
        return data.astype(np.float32)
    else:
        if DATA_MODE == 'shaking_table':
            df = pd.read_csv(fpath, sep=r'\s+', header=None, engine='c')
            return df.iloc[:, -3:].values.astype(np.float32)
            
        clean_lines = []
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                lower_line = line.strip().lower()
                if lower_line.startswith('location') or lower_line.startswith('unknown') or lower_line == '':
                    continue
                clean_lines.append(line)
        if len(clean_lines) < FS * 3: 
            return None
        df = pd.read_csv(io.StringIO(''.join(clean_lines)), sep=r'\s+', header=None, engine='python', on_bad_lines='skip')
        return df.iloc[:, -3:].values.astype(np.float32)

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
            
            if not self.triggered: curr_lta = self.sum_lta / self.nlta
            else: curr_lta = self.frozen_lta
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
                    
                    points_to_packet_end = (int(FS) - (i % int(FS))) % int(FS)
                    if points_to_packet_end == 0: 
                        points_to_packet_end = int(FS) 
                    
                    self.cp_packet_1 = points_to_packet_end                 
                    self.cp_packet_2 = points_to_packet_end + int(1.0 * FS) 
                    
                    self.tracking_timer = 0
                    self.event_decided = False
                    self.shutdown_counter = 0

            else:
                if not self.event_decided:
                    self.tracking_timer += 1
                    
                    if self.tracking_timer == self.cp_packet_1:
                        trigger_idx = i - self.cp_packet_1
                        is_eq = self._run_cnn_tracking_only(i, trigger_idx, events, filename, is_early_exit=True)
                        if is_eq:
                            self.event_decided = True 

                    elif self.tracking_timer == self.cp_packet_2:
                        trigger_idx = i - self.cp_packet_2
                        self._run_cnn_tracking_only(i, trigger_idx, events, filename, is_early_exit=False)
                        self.event_decided = True 

                if ratio < THRESHOLD_OFF:
                    self.shutdown_counter += 1
                    if self.shutdown_counter > int(SHUTDOWN_WAIT_SEC * FS):
                        self.triggered = False
                        self.cooldown_counter = int(COOLDOWN_SEC * FS)
                        self.shutdown_counter = 0
                        if events and events[-1]['p_time_s'] > 0: events[-1]['event_end_idx'] = i
                else:
                    self.shutdown_counter = 0
                    
        return ratio_history, filt_ud_history, events

    def _run_cnn_tracking_only(self, current_stream_idx, trigger_idx, events_list, filename, is_early_exit=False):
            buffer_arr = np.array(self.raw_data_buffer, dtype=np.float32)
            buf_end = len(buffer_arr)
            win_pts = CNN_INPUT_PTS 
            step_pts = int(SLIDE_STEP_SEC * FS)
            
            search_start_idx = buf_end - int((SEARCH_POST_SEC + SEARCH_PRE_SEC) * FS)
            if search_start_idx < 0: return False
            search_end_idx = buf_end
            
            batch_clips = []
            batch_start_indices = []
            
            curr = search_start_idx
            while curr + win_pts <= search_end_idx:
                clip = buffer_arr[curr : curr + win_pts]
                clip_demean = clip - np.mean(clip, axis=0)
                max_val = np.max(np.abs(clip_demean))
                clip_norm = clip_demean / max_val
                batch_clips.append(clip_norm)
                batch_start_indices.append(curr)
                curr += step_pts
                
            if not batch_clips: return False
            
            batch_clips = np.array(batch_clips, dtype=np.float32)
            preds = self.model.predict(batch_clips, verbose=0)
            probs_eq = preds[0][:, 1]
            pick_curves = preds[1]
            
            best_candidate = None
            best_score = -1.0
            
            for i in range(len(batch_clips)):
                prob = float(probs_eq[i])
                if prob < 0.3: continue 
                
                p_rel_idx = np.argmax(pick_curves[i])
                p_buf_idx = batch_start_indices[i] + p_rel_idx
                
                buffer_trigger_rel_idx = buf_end - (current_stream_idx - trigger_idx)
                dt = (p_buf_idx - buffer_trigger_rel_idx) / FS 
                
                if -4.0 <= dt <= 0.2: time_weight = 1.0
                elif dt > 0.2: time_weight = 0.1 
                else: time_weight = 0.5
                
                score = prob * time_weight
                
                if score > best_score:
                    best_score = score
                    best_candidate = {'prob': prob, 'p_buf_idx': p_buf_idx, 'dt': dt}
            
            if best_candidate and best_candidate['prob'] >= EQ_PROB_THRESHOLD and best_score > 0.2:
                offset_from_end = best_candidate['p_buf_idx'] - buf_end
                p_stream_idx = current_stream_idx + offset_from_end
                events_list.append({
                    'file': filename, 'event_type': 'earthquake', 'reject_reason': 'None',
                    'trigger_time_s': trigger_idx / FS, 'trigger_time_idx': trigger_idx,
                    'p_time_s': p_stream_idx / FS, 'p_prob': best_candidate['prob'],
                    'trigger_to_p_delay_s': best_candidate['dt'],
                    'event_start_idx': self.current_event_start, 'event_end_idx': current_stream_idx
                })
                return True 
                
            else:
                if not is_early_exit:
                    events_list.append({
                        'file': filename, 'event_type': 'noise',
                        'reject_reason': '3_low_CNN_score' if (best_candidate and best_candidate['prob']<EQ_PROB_THRESHOLD) else '3_Error_time',
                        'trigger_time_s': trigger_idx / FS, 'trigger_time_idx': trigger_idx,
                        'p_time_s': -1, 'p_prob': best_candidate['prob'] if best_candidate else 0.0,
                        'trigger_to_p_delay_s': best_candidate['dt'] if best_candidate else -999,
                        'event_start_idx': self.current_event_start, 'event_end_idx': current_stream_idx
                    })
                return False


def plot_stream_result(filename, raw_ud, filt_ud, ratio, events, save_dir, true_p_idx=None, title_prefix=""):

    plt.close('all')
    
    times = np.arange(len(raw_ud)) / FS
    
    eq_events = [e for e in events if e['event_type'] == 'earthquake']
    n_eq = len(eq_events)

    total_rows = 2 + n_eq
    fig_height = 10 + n_eq * 2.5
    
    fig = plt.figure(figsize=(16, fig_height))

    gs = gridspec.GridSpec(total_rows, 1, figure=fig, hspace=0.35)

    #fig1 wave
    ax1 = fig.add_subplot(gs[0, 0])
    full_title = f"{title_prefix}: {filename} | EQ Found: {n_eq}"
    ax1.set_title(full_title, fontsize=20, fontweight='bold',pad=35)
    ax1.plot(times, raw_ud, 'lightgray', lw=0.8, label='Raw UD')
    ax1.plot(times, filt_ud, 'k', lw=0.8, label='Filtered UD')
    ax1.set_ylabel("Acc ($m/s^2$)", fontsize=18) 
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(times[0], times[-1]) 

    ax1.tick_params(axis='both', labelsize=16)
    
    # fig2 sta/lta Ratio 
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(times, ratio, 'g', lw=1, label='STA/LTA Ratio')
    ax2.axhline(THRESHOLD_ON, color='r', ls='--', label='On Threshold')
    ax2.axhline(THRESHOLD_OFF, color='orange', ls=':', label='Off Threshold')
    ax2.set_ylabel("Ratio", fontsize=18)
    ax2.grid(True, alpha=0.3)

    ax2.tick_params(axis='both', labelsize=16)
    
    # true P (for continuous data)
    has_true_p = (true_p_idx is not None and 0 <= true_p_idx < len(raw_ud))
    if has_true_p:
        true_p_time = true_p_idx / FS
        ax1.axvline(true_p_time, color='green', lw=2.0, alpha=0.8)


    for i, evt in enumerate(events):
        trig = evt['trigger_time_s']
        is_eq = (evt['event_type'] == 'earthquake')
        c = 'r' if is_eq else 'gray'
        a = 0.15 if is_eq else 0.05

        if 'event_start_idx' in evt:
            t_s = evt['event_start_idx'] / FS
            t_e = evt['event_end_idx'] / FS
            ax1.axvspan(t_s, t_e, color=c, alpha=a)
            ax2.axvspan(t_s, t_e, color=c, alpha=a)

        ax1.axvline(trig, color='blue', ls=':', alpha=0.5,label='STA/LTA Trigger')
        ax2.axvline(trig, color='blue', ls=':', alpha=0.5,label='STA/LTA Trigger' )
    
        
        if is_eq and evt['p_time_s'] > 0:
            p_t = evt['p_time_s']
            prob = evt['p_prob']
            ax1.axvline(p_t, color='red', lw=1.5, ls='--',label='Pred P')
            y_max = np.percentile(np.abs(filt_ud), 99.5) if len(filt_ud) > 0 else 1
            bbox_props = dict(boxstyle="round,pad=0.3", fc="yellow", ec="red", alpha=0.8)
            ax1.text(p_t-1, y_max, f"P at {p_t:.2f}s\nProb={prob:.2f}", color='darkred', fontsize=14, 
                     fontweight='bold', ha='center', va='bottom', bbox=bbox_props)

    def make_unique_legend(ax, loc='upper right', ncol=1):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        priority = ['Raw UD', 'Filtered UD', 'True P', 'Pred P', 'STA/LTA Trigger', 'STA/LTA Ratio']
        sorted_keys = sorted(by_label.keys(), key=lambda x: priority.index(x) if x in priority else 99)
        sorted_handles = [by_label[k] for k in sorted_keys]
        ax.legend(sorted_handles, sorted_keys, loc=loc, ncol=ncol, fontsize=16)

    make_unique_legend(ax1, ncol=2)
    make_unique_legend(ax2, ncol=2)
    
    for i, evt in enumerate(eq_events):
        p_t = evt['p_time_s']
        trig_t = evt['trigger_time_s']
        prob = evt['p_prob']
        
        plot_start_t = max(0, p_t - 3.0)
        plot_end_t = min(times[-1], p_t + 3.0)
        
        idx_s = int(plot_start_t * FS)
        idx_e = int(plot_end_t * FS)
        
        zoom_time = times[idx_s:idx_e]
        zoom_data = filt_ud[idx_s:idx_e] 
        
        ax_zoom = fig.add_subplot(gs[2+i, 0])
        ax_zoom.plot(zoom_time, zoom_data, 'k', lw=1.2, label='Filtered')
    
        ax_zoom.axvline(p_t, color='red', lw=2.0, ls='--', label='Pred P')
        ax_zoom.axvline(trig_t, color='blue', lw=1.5, ls=':', label='STA/LTA Trigger')
        
        if has_true_p:
            true_p_time = true_p_idx / FS
            if plot_start_t <= true_p_time <= plot_end_t:
                ax_zoom.axvline(true_p_time, color='green', lw=2.5, alpha=0.6, label='True P')
        
        ax_zoom.set_title(f"Event #{i+1} Zoom (P at {p_t:.2f}s | Prob={prob:.2f})", fontsize=18, color='darkred')
        ax_zoom.set_ylabel("Acc", fontsize=18)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.set_xlim(plot_start_t, plot_end_t)
        ax_zoom.tick_params(axis='both', labelsize=16)

        ax_zoom.legend(loc='upper left', fontsize=16, framealpha=0.8)

    plt.xlabel("Time (s)", fontsize=18)
    plt.subplots_adjust(top=0.95)
    
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

def get_max_pa(ud_filt_data, start_i, end_i):
    if pd.isna(start_i): return pd.NA
    s = int(max(0, start_i))
    e = int(min(len(ud_filt_data), end_i))
    if s >= e: return pd.NA
    clip = ud_filt_data[s:e]
    clip_demean = clip - np.mean(clip)
    pa_val = float(np.max(np.abs(clip_demean)))
    return round(pa_val, 2)


def run_unified_pipeline(detector):
    if not os.path.exists(INPUT_DIR):
        print(f"[Error] Input directory not found: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    MISSED_PLOT_DIR = os.path.join(OUTPUT_DIR, "Missed_files")
    os.makedirs(MISSED_PLOT_DIR, exist_ok=True)
    CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "processing_checkpoint.csv")


    if os.path.exists(CHECKPOINT_FILE):
        print(f"recover from {CHECKPOINT_FILE}")
        df = pd.read_csv(CHECKPOINT_FILE)
    else:
        all_files = glob.glob(os.path.join(INPUT_DIR, "**", "*.[nt]*"), recursive=True)
        all_files = [f for f in all_files if f.endswith('.npy') or f.endswith('.txt')]
        
        if not all_files:
            print(f"[Warning] No valid trace files (.npy or .txt) found in {INPUT_DIR}")
            return
            
        records = []
        for f in all_files:
            fname = os.path.basename(f)
            pidx = pd.NA

            if '_pidx_' in fname:
                try: pidx = int(fname.split('_pidx_')[-1].split('.')[0])
                except: pass
            
            records.append({
                "currenten_fpath": f, 
                "idx": pidx, 
                "year": "SampleData", 
                "comp": "UD"
            })
            
        df = pd.DataFrame(records)
        df["pre_class"] = pd.NA
        df["pre_pidx"] = pd.NA
        df["pre_prob"] = pd.NA
        df["status"] = pd.NA  
        df["true_pa_1s"] = pd.NA  
        df["true_pa_3s"] = pd.NA  
        df["trig_pa_1s"] = pd.NA  
        df["trig_pa_3s"] = pd.NA  
        df["max_sta_lta"] = pd.NA 

    total_samples = len(df)
    print(f"total files: {total_samples}")
    
    count_skipped_short = (df["status"] == "short").sum() if "status" in df else 0
    count_skipped_error = (df["status"] == "error").sum() if "status" in df else 0
    count_processed = (df["status"] == "processed").sum() if "status" in df else 0
    
    if count_processed > 0 or count_skipped_short > 0 or count_skipped_error > 0:
        print(f"skip: {count_skipped_short}")
        print(f"Error: {count_skipped_error}")
        print(f"Success: {count_processed}")
        print(f"Waiting for processing: {total_samples - count_skipped_short - count_skipped_error - count_processed}\n")

    MIN_DATA_LEN_PTS = int(FS * 5.5) 
    newly_processed = 0 
    missed_analysis_list = []
    

    for i, row in df.iterrows():
        if pd.notna(row.get("status")): continue
            
        raw_path = row["currenten_fpath"]
        try:
            raw_data = load_trace_data(raw_path)
            if raw_data is None or len(raw_data) < MIN_DATA_LEN_PTS:
                count_skipped_short += 1
                df.at[i, "status"] = "short"
                continue 
            
            detector.reset()
            warmup_len = min(len(raw_data), int(3.0*FS))
            init_chunk = raw_data[:warmup_len]
            if len(init_chunk) > 0:
                noise_lvl = np.median(np.abs(init_chunk[:, 2]))
                if noise_lvl < 1e-10: noise_lvl = 1e-10
                detector.warm_up(noise_level=noise_lvl, duration_sec=15.0)
            
            filename = os.path.basename(raw_path)
            ratio, filt, events = detector.process_data_stream(raw_data, filename)
            
            max_ratio_val = float(np.max(ratio)) if len(ratio) > 0 else 0.0
            df.at[i, "max_sta_lta"] = round(max_ratio_val, 2)

            valid_eq_candidates = [e for e in events if e['event_type'] == 'earthquake' and e['p_time_s'] >= 0]
            prediction_class = 1 if len(valid_eq_candidates) > 0 else 0
            final_prob, final_pidx = 0.0, -1
            matched_evt = None 

            true_idx = row.get("idx")
            if pd.notna(true_idx):
                df.at[i, "true_pa_1s"] = get_max_pa(filt, true_idx, true_idx + int(1.0 * FS))
                df.at[i, "true_pa_3s"] = get_max_pa(filt, true_idx, true_idx + int(3.0 * FS))

            if prediction_class == 1:
                target_idx = row["idx"]
                if pd.notna(target_idx):
                    min_dist = float('inf')
                    for evt in valid_eq_candidates:
                        pred_idx = int(round(evt['p_time_s'] * FS))
                        dist = abs(pred_idx - target_idx)
                        if dist < min_dist:
                            min_dist = dist
                            matched_evt = evt
                else:
                    matched_evt = valid_eq_candidates[0]
                
                if matched_evt:
                    final_prob = matched_evt['p_prob']
                    final_pidx = int(round(matched_evt['p_time_s'] * FS))
                    trig_idx = int(matched_evt['trigger_time_idx'])
                    df.at[i, "trig_pa_1s"] = get_max_pa(filt, trig_idx, trig_idx + int(1.0 * FS))
                    df.at[i, "trig_pa_3s"] = get_max_pa(filt, trig_idx, trig_idx + int(3.0 * FS))
            else:
                if len(events) > 0:
                    first_trig_idx = int(events[0]['trigger_time_idx'])
                    df.at[i, "trig_pa_1s"] = get_max_pa(filt, first_trig_idx, first_trig_idx + int(1.0 * FS))
                    df.at[i, "trig_pa_3s"] = get_max_pa(filt, first_trig_idx, first_trig_idx + int(3.0 * FS))

            df.at[i, "pre_class"] = prediction_class
            df.at[i, "pre_prob"] = final_prob
            df.at[i, "pre_pidx"] = final_pidx
            df.at[i, "status"] = "processed"
            count_processed += 1
            newly_processed += 1

            if prediction_class == 0:
                if len(events) == 0:
                    reason = "1_STA/LTA_No_trigger"
                else:
                    reached_cnn = any('trigger_to_p_delay_s' in e and e.get('trigger_to_p_delay_s') != -999 for e in events)
                    if reached_cnn: reason = "3_low_CNN_score"
                    else: reason = next((e.get('reject_reason') for e in events if 'reject_reason' in e), "2_gating")
                
                missed_analysis_list.append({
                    'year': row.get('year', 'SampleData'),
                    'comp': row.get('comp', 'UD'),
                    'origi': filename,
                    'missed reasons': reason,
                    'max STA_LTA ratio': round(max_ratio_val, 2)
                })

            if ENABLE_PLOTTING:
                if prediction_class == 0:
                    plot_stream_result(filename, raw_data[:, 2], filt, ratio, events, MISSED_PLOT_DIR, true_p_idx=row["idx"], title_prefix="MISSED EQ")
                else:
                    save_dir = os.path.join(OUTPUT_DIR, "Detect_Pics")
                    plot_stream_result(filename, raw_data[:, 2], filt, ratio, events, save_dir, true_p_idx=row["idx"], title_prefix="HIT EQ")

            if newly_processed > 0 and newly_processed % 50 == 0:
                print(f"Processed {i+1}/{total_samples}")
                df.to_csv(CHECKPOINT_FILE, index=False)

        except Exception as e:
            print(f"Error {raw_path}: {e}")
            count_skipped_error += 1
            df.at[i, "status"] = "error"

    df.to_csv(CHECKPOINT_FILE, index=False)
    print("\nDone")

    print("\n" + "="*50)
    print("            EARTHQUAKE DIAGNOSTIC REPORT             ")
    print("="*50)
    print(f"[1. Data Flow]")
    print(f"  Total Samples: {total_samples}")
    print(f"  Successfully Processed:     {count_processed}")
    print(f"  SKIPPED (Too Short < 5.5s): {count_skipped_short}")
    print(f"  SKIPPED (File Error):       {count_skipped_error}")
    
    if count_processed == 0: return

    valid_df = df[df["status"] == "processed"].copy()
    n_eq_total = len(valid_df)
    n_eq_correct = len(valid_df[valid_df["pre_class"] == 1]) 
    n_eq_missed = len(valid_df[valid_df["pre_class"] == 0])  
    
    if n_eq_total > 0:
        print(f"\n[2. Detection Performance]")
        print(f"  Detected (TP):           {n_eq_correct}  (Recall: {n_eq_correct/n_eq_total:.2%})")
        print(f"  Missed (FN):             {n_eq_missed}  (Miss Rate: {n_eq_missed/n_eq_total:.2%})")
        
        missed_df = valid_df[valid_df["pre_class"] == 0]
        missed_csv_path = os.path.join(OUTPUT_DIR, "missed_earthquakes_list.csv")
        missed_df.to_csv(missed_csv_path, index=False)
        print(f"\n[3. Error Recordings]")
        print(f" Missed files: {missed_csv_path}")
        print(f" Missed pics: {MISSED_PLOT_DIR}")
        
        if n_eq_correct > 0 and valid_df["idx"].notna().sum() > 0:
            tp_df = valid_df[(valid_df["pre_class"] == 1) & (valid_df["idx"].notna())].copy()
            if len(tp_df) > 0:
                tp_df["pre_pidx"] = pd.to_numeric(tp_df["pre_pidx"], errors='coerce')
                tp_df["idx"] = pd.to_numeric(tp_df["idx"], errors='coerce')
                
                tp_df["p_error_abs"] = (tp_df["pre_pidx"] - tp_df["idx"]).abs()
                errors_sec_signed = (tp_df["pre_pidx"] - tp_df["idx"]) / FS
                errors_sec_abs = tp_df["p_error_abs"] / FS
                
                p_hit_05s = (tp_df["p_error_abs"] <= 50).mean()
                p_hit_1s = (tp_df["p_error_abs"] <= 100).mean()
                
                print(f"\n[4. P-Pick Accuracy]")
                print(f"  P-hit (0.5s):            {p_hit_05s:.4f}")
                print(f"  P-hit (1.0s):            {p_hit_1s:.4f}")
                
                print(f"\n[5. Error Distribution Statistics (Seconds)]")
                print(f"  Mean Error:              {errors_sec_signed.mean():.4f}")
                print(f"  Std Deviation:           {errors_sec_signed.std():.4f}")
                print(f"  Mean Absolute Error:     {errors_sec_abs.mean():.4f}")
                print(f"  Min Error:               {errors_sec_signed.min():.4f}")
                print(f"  Max Error:               {errors_sec_signed.max():.4f}")
                print(f"  25th Percentile (Q1):    {errors_sec_signed.quantile(0.25):.4f}")
                print(f"  Median (Q2):             {errors_sec_signed.median():.4f}")
                print(f"  75th Percentile (Q3):    {errors_sec_signed.quantile(0.75):.4f}")
                
                plot_p_error_distribution(tp_df, OUTPUT_DIR)
            
    print("="*50)

    if missed_analysis_list:
        df_miss_details = pd.DataFrame(missed_analysis_list)
        stats = df_miss_details['missed reasons'].value_counts().reset_index()
        stats.columns = ['Missed Phases', 'Cases Num']
        stats['Proportion'] = (stats['Cases Num'] / stats['Cases Num'].sum()).apply(lambda x: f"{x:.2%}")
        stats = stats.sort_values(by='Missed Phases') 
        
        stats_csv_path = os.path.join(OUTPUT_DIR, "Missed_EQ_Analysis_1_whole.csv")
        details_csv_path = os.path.join(OUTPUT_DIR, "Missed_EQ_Analysis_2_details.csv")
        
        stats.to_csv(stats_csv_path, index=False, encoding='utf-8-sig')
        df_miss_details.to_csv(details_csv_path, index=False, encoding='utf-8-sig')
        
        print(f"Whole Cases: {stats_csv_path}")
        print(f"Details: {details_csv_path}")

def main():
    print(f"[INFO] Operational Mode Selected: [{DATA_MODE.upper()}]")
    detector = OptimizedStreamDetector()
    run_unified_pipeline(detector)

if __name__ == "__main__":
    main()