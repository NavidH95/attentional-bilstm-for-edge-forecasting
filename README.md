# Jena Climate Time Series Forecasting

This project implements a Bidirectional LSTM with Attention model to forecast weather data from the Jena Climate dataset. The model is trained using PyTorch and can be converted to TFLite for efficient inference.

## Project Structure
```
├── data/                  # Contains the raw CSV data
├── models/                # Python files for model architecture
├── train/                 # Scripts for training and evaluating the model
├── inference/             # Scripts for running inference with trained models
├── outputs/               # Directory for all generated files (models, plots)
├── requirements.txt       # Python dependencies
└── README.md              # Project explanation
```
## How to Run

### 1. Setup

First, create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 2. Download Data
Place the `jena_climate_2009_2016.csv` file inside the `data/` directory.

### 3. Training
To train the model, run the main training script. This will process the data, train the model, save the best version, and generate evaluation plots.

```bash
python train/train.py
```

The trained model (`final_best_model.pth`), scaler (`scaler_params.npz`), and plots will be saved in the `outputs/` directory.

### 4. Convert to TFLite
After training, you can convert the saved PyTorch model to a TFLite model for efficient deployment.

```bash
python inference/convert_to_tflite.py
```

The `model.tflite` file will be created in `outputs/saved_models/`.

### 5. Run TFLite Inference
To test the TFLite model with a random sample, run:

```bash
python inference/run_inference.py
```
