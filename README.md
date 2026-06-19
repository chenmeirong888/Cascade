# Smartphone-Edge Cascading Earthquake Early Warning (EEW) Pipeline

This repository contains the open-source Python implementation of the cascading framework for smartphone-based Earthquake Early Warning (EEW) systems, as described in our manuscript.
The processing pipeline operates on stream telemetrics in three distinct stages:
1. **Front-End**: STA/LTA continuous filtering and initial trigger activation.
2. **Mid-End**: A 4-Stage Physical Feature Gating layer to filter out non-seismic human operational noise.
3. **Back-End**: Sliding convolutional neural network (CNN) inference for event classification and precise P-wave phase picking.

---

## Repository Structure

```text
├── data_input/
│   ├── phone_records/
│   │   ├── human_noise/             # Daily human noise records (5-column format)
│   │   └── shaking_table/           # Shaking table benchmark files (3-column pure acceleration matrix)
│   ├── seismic/                     # Long-term continuous seismic waveforms 
│   └── earthquake.db                # SQLite3 database containing indexed seismogram metadata
│
│
├── data_output/                     # Output directories for evaluation reports
│   ├── human_noise/
│   ├── shaking_table/
│   └── continuous_eq/
│
├── models/
│       └── fold_0_v4.h5             # Fully fine-tuned pre-trained CNN weights used in the paper
│
├── run_prediction.py                # Pipeline execution and evaluation script
└── train.py                         # Model architecture and training source code

Environmental Prerequisites
Ensure you have Python 3.8+ installed along with the stable computing and software stack

Execution & Evaluation
The workspace configuration can be toggled via the DATA_MODE parameter at the top of prediction.py:

# Open prediction.py and specify the evaluation mode:
DATA_MODE = 'human_noise'      # For daily human operational noise benchmarks
# DATA_MODE = 'shaking_table'   # For pure laboratory shaking table benchmarks
# DATA_MODE = 'continuous_eq'   # For long-term continuous seismogram evaluation via SQLite3

Run the pipeline using standard execution:
python prediction.py

Generated Outputs:
1-Statistical Summaries: Benchmarking summaries (e.g., summary_statistics.csv, Missed_EQ_Diagnostic_Details.csv) are written directly to the corresponding data_output/ directory.
2-Time-series probability alignment graphs plotting Predicted P-waves (dashed red line) and True P-waves (dashed green line) are exported to disk when plotting parameters are enabled.

Dataset Notice
1. Inference Benchmark: Users can run run_prediction.py directly using the enclosed micro-telemetry data fragments to verify the core logic and workflow described in the manuscript.
2. Model Training: The full-scale telemetry catalogs and historical continuous seismic waveform databases used for the 5-fold cross-validation are omitted from this repository due to storage limitations. The train.py script is provided as a methodological reference for peer-review audit.

Citation
If you find this codebase or our framework useful for your academic research, please consider citing our work.
