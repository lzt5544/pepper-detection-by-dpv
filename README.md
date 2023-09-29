# Pepper-detection-by-dpv

## Overview 

This repository is dedicated to training models for differentiating the origin, identifying adulteration, and predicting pungency intensity of Sichuan pepper using electrochemical voltammetry data.

## Requirements

* Python 3.9+
* Works on Linux, Windows, macOS

## Getting Started

### Installation

* Clone this repo:
```bash
    git clone https://github.com/lzt5544/pepper-detection-by-dpv.git
    cd pepper-detection-by-dpv
```
* Install dependencies:
```bash
    pip install -r requirements.txt
```

### Train & Validation

* Train
```bash
    python main.py -m svm knn xgb ann -c config/config_pungency.yml -t
```

* Validation
```bash
    python main.py -m svm knn xgb ann -c config/config_pungency.yml -v
```

The training results and models are saved in result folder