# A Cascaded Earthquake Detection and Phase Picking Algorithm for Smartphone-Based Early Warning Systems

This repository contains the open-source Python implementation of the cascading framework for smartphone-based Earthquake Early Warning (EEW) systems, as described in our manuscript.
The processing pipeline includes:
1. **Front-End**: STA/LTA filtering and initial trigger activation.
2. **Back-End**: Convolutional Neural Network (CNN) inference for event classification and P-wave phase picking.

---

## Repository Structure

```text
├── data_samples/ # Demonstration datasets
│   ├── 1_continuous data/ # Continuous stream records with true P-wave indices (_pidx_ tags)
│   ├── 2_shaking_table/ # Shaking table benchmark files (3-column pure acceleration matrix)
│   ├── 3_human_daily/  # Daily human noise records (5-column format)
│   ├── 4_train_5s/  # 5s-window segments for CNN training
│   ├── 5_data_20s/ #20s-window seismic segments 
│
├── paper_cnn_model/                 #  CRITICAL: Contains the fine-tuned weights used in the paper
├── train_model/                     # Output directory for trained CNN weights
├── fine_tune_model/                 # Output directory for trained CNN weights
├── pred_output/                     # Output directory for evaluation reports
│
├── data_slide_20s.py          # Preprocessing: Raw -> 20s segments
├── slide_5s.py                # Preprocessing: 20s -> 5s training segments
├── train.py                   # Model training script
├── finetune.py                # Fine-tuning script with partial layer-freezing
└── prediction.py              # Real-time stream inference & evaluation
│
├── requirements.txt          
```

**Pipeline Workflow**
1. Data Slide
- 20s Extraction (data_slide_20s.py): Scans 1_continuous data/ and extracts 20-second seismic clips. It dynamically calculates the true P-wave index relative to the sliding window and embeds this true location into the output .npy filename (e.g., _pidx_850.npy).
- 5s Segmentation (slide_5s.py): Reads the 20s segments and generates 5-second training arrays stored in data_samples/4_train_5s/EQ/.

2. Model Training & Fine-Tuning
- Run train.py to train the CNN model.
- The training process automatically reads segments from 4_train_5s/. A dynamic resampling strategy is employed during training to maintain a 1:1 ratio between positive and negative samples per batch, mitigating class imbalance. Output weights are saved into the train_model/ directory.
- finetune.py implements a domain adaptation strategy by freezing the bottom 60% of convolutional blocks to prevent catastrophic forgetting.

3. Inference & Evaluation
The inference pipeline operates on three distinct modes defined by the DATA_MODE variable in prediction.py:
- 'human_noise': Evaluates precision on daily activity datasets.
- 'shaking_table': Evaluates performance on laboratory shaking scenarios.
- 'continuous_eq': Evaluates recall and P-wave picking accuracy on continuous seismic streams.


**Environmental Prerequisites**
- Ensure you have Python 3.8+ installed along with the stable computing and software stack
- numpy, pandas,scipy, matplotlib,scikit-learn


**Quick Start**

To run the evaluation pipeline:
1. Open prediction.py.
2. Set DATA_MODE to your desired configuration (e.g., 'continuous_eq').
3. Execute: python prediction.py


**Generated Outputs**
- Evaluation Reports: CSV summaries (e.g., summary_statistics.csv, missed_earthquakes_list.csv) are written directly to the corresponding pred_output/DATA_MODE/ directory.
- Diagnostic Visualizations: Time-series probability alignment graphs (showing raw/filtered signals, STA/LTA ratios, and P-wave predictions) are exported as PNGs for audit.


**Data and Reproducibility Notice**
1. Inference Reproduction: To reproduce the evaluation metrics and figures reported in the manuscript, simply execute prediction.py. This script defaults to using our final fine-tuned model weights (/paper_cnn_model/fold_0_v4.h5).
2. Dataset Limitations: Due to file size constraints and ongoing research embargoes on the full-scale continuous seismic telemetry, this repository only includes a representative data subset (data_samples/).
3. Methodological Reference: The provided processing and training scripts (data_slide_20s.py, train.py, finetune.py) are included to demonstrate the algorithmic workflow. Local training results generated using this limited subset are intended for methodological reference rather than production-level performance.


**Citation**

If you find this codebase or our framework useful for your academic research, please consider citing our work.
