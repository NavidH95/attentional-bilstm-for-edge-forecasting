import torch
import ai_edge_torch
import numpy as np
import os

# Import model definition
from models.bilstm_attention import BiLSTMAttentionModel

# --- Configuration ---
PYTORCH_MODEL_PATH = "outputs/saved_models/final_best_model.pth"
TFLITE_MODEL_PATH = "outputs/saved_models/model.tflite"

# Model Hyperparameters (must match the trained model)
INPUT_DIM = 14
OUTPUT_DIM = 14
HIDDEN_DIM = 64
NUM_LAYERS = 2
SEQUENCE_LENGTH = 128

# --- Main Script ---
if __name__ == "__main__":
    if not os.path.exists(PYTORCH_MODEL_PATH):
        print(f"Error: PyTorch model not found at {PYTORCH_MODEL_PATH}")
        print("Please run the training script first: python train/train.py")
    else:
        # Load the PyTorch model
        model = BiLSTMAttentionModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM)
        model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location='cpu'))
        model.eval()

        # Create a sample input for conversion
        sample_input = (torch.randn(1, SEQUENCE_LENGTH, INPUT_DIM),)

        # Convert the model using ai-edge-torch
        edge_model = ai_edge_torch.convert(model, sample_input)
        edge_model.export(TFLITE_MODEL_PATH)

        print(f"Model successfully converted and saved to '{TFLITE_MODEL_PATH}'")