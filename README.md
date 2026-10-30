
# Zero-Trust IoT Research

## Overview

This repository contains the Python implementation of my BSc Honours in Computer Science research project at the University of Fort Hare.

**Research title:** Machine Learning and Risk-Based Trust Evaluation for Continuous Authentication in IoT Zero-Trust Systems

## Research Focus

This project investigates risk-based trust evaluation for Internet of Things (IoT) network traffic within a Zero-Trust environment. It explores how machine-learning models can identify potentially malicious network activity and estimate risk to support trust decisions during continuous authentication.

The research focuses on evaluating model performance, examining generalisation to different datasets, and investigating how risk assessments can inform trust decisions over time.

## Implementation Pipeline

The implementation is organised into numbered Python scripts covering the following stages:

1. **Raw data inventory and loading** (`0_raw_data_inventory.py`, `1_data_loading.py`) — inspecting and loading network traffic datasets.
2. **Data cleaning and preparation** (`2_data_cleaning.py`) — preparing data for analysis and modelling.
3. **Data splitting and validation design** (`3_train_test_split.py`, `4_validation_design.py`) — establishing training, testing, and validation procedures.
4. **Feature selection and exploratory data analysis** (`5_feature_selection.py`, `6_eda.py`) — examining dataset characteristics and selecting relevant features.
5. **Generalisation diagnostics** (`5b_generalisation_diagnostics.py`) — investigating model behaviour across data sources and evaluation settings.
6. **Model tuning and risk modelling** (`7_tuning.py`, `8_risk_models_and_hybrid.py`) — developing and evaluating risk estimation models.
7. **Trust decision layer** (`9_decision_layer.py`) — translating risk estimates into trust-related decisions.
8. **Trust lifecycle evaluation** (`10_trust_lifecycle.py`) — examining changes in trust status over time.
9. **Final evaluation** (`11_final_evaluation.py`) — assessing the overall performance of the proposed approach.
10. **Trust-level visualisation** (`12_trust_level_figure.py`) — generating a figure comparing the distribution of benign and malicious test flows across the Trusted, Suspicious, and Untrusted levels for RT-IoT2022 and IoT-23. The script uses existing evaluation results and produces a PNG figure, a CSV containing the figure data, and source hashes.

The implementation and evaluation procedures may be refined as the research progresses.

## Technologies

- **Programming language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, XGBoost
- **Visualisation:** Matplotlib
- **Model explainability:** SHAP

## Repository Structure

```text
Zero_Trust_Research/
├── config/
│   └── feature_selection_protocol.json
├── docs/
│   └── D01_feature_selection_protocol.md
├── src/
│   ├── 0_raw_data_inventory.py
│   ├── 1_data_loading.py
│   ├── 2_data_cleaning.py
│   ├── 3_train_test_split.py
│   ├── 4_validation_design.py
│   ├── 5_feature_selection.py
│   ├── 5b_generalisation_diagnostics.py
│   ├── 6_eda.py
│   ├── 7_tuning.py
│   ├── 8_risk_models_and_hybrid.py
│   ├── 9_decision_layer.py
│   ├── 10_trust_lifecycle.py
│   ├── 11_final_evaluation.py
│   └── 12_trust_level_figure.py
├── .gitignore
├── README.md
└── requirements_lock.txt
```

## Setup

### Prerequisites

- Python 3.12
- pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/CJ-Mampane/Zero_Trust_Research.git
   cd Zero_Trust_Research
   ```

2. Create a virtual environment:

   **Windows:**
   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\python.exe -m pip install --upgrade pip
   ```

   **Linux/macOS:**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```

3. Install the dependencies:

   **Windows:**
   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements_lock.txt
   ```

   **Linux/macOS:**
   ```bash
   python -m pip install -r requirements_lock.txt
   ```

Individual scripts may require additional configuration or dataset paths before they can be executed. Follow any relevant instructions in the source code and documentation.

## Data and Reproducibility

Datasets, trained models, generated results, and local development environments are excluded from this repository.

To reproduce the experiments, obtain the required datasets from their original sources, review the applicable licences and usage conditions, and configure the relevant paths and settings. Follow the documented data preparation, splitting, validation, and evaluation procedures when reproducing results.

## Research Status

This repository represents an ongoing academic research project. The implementation, experimental procedures, and evaluation may change as the research develops. Results and conclusions should be interpreted in the context of the current implementation and documented experimental design.

## Author

**Cathrene Jabulile Mampane**  
BSc Honours in Computer Science  
Faculty of Computational Sciences  
University of Fort Hare, South Africa
