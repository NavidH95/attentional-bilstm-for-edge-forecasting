import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Import model from the models folder
from models.bilstm_attention import BiLSTMAttentionModel
from models.lstm_attention import LSTMAttentionModel
from models.bilstm import BiLSTMModel
from models.lstm import LSTMModel


# --- Configuration ---
# Paths
DATA_PATH = "data/jena_climate_2009_2016.csv"
MODEL_SAVE_PATH = "outputs/saved_models/final_best_model.pth"
SCALER_SAVE_PATH = "outputs/saved_models/scaler_params.npz"
PLOT_SAVE_PATH = "outputs/plots/"

# Create directories if they don't exist
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(PLOT_SAVE_PATH, exist_ok=True)

# Hyperparameters
SEQUENCE_LENGTH = 128
STRIDE = SEQUENCE_LENGTH // 2
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 64
N_SPLITS = 5

# --- Helper Functions ---
def create_sequences_with_overlap(data, sequence_length, stride):
    X, y = [], []
    for i in range(0, len(data) - sequence_length, stride):
        seq_x = data[i:i + sequence_length]
        seq_y = data[i + sequence_length]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def evaluate_epoch(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
    return running_loss / len(data_loader.dataset)


# --- Main Script ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading and Preprocessing
    df = pd.read_csv(DATA_PATH).drop("Date Time", axis=1)
    feature_names = df.columns.tolist()

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    np.savez(SCALER_SAVE_PATH, min=scaler.min_, scale=scaler.scale_)
    print(f"Scaler parameters saved to '{SCALER_SAVE_PATH}'")

    X_data, y_data = create_sequences_with_overlap(scaled_data, SEQUENCE_LENGTH, STRIDE)
    X_tensor = torch.from_numpy(X_data).float()
    y_tensor = torch.from_numpy(y_data).float()

    INPUT_DIM = X_tensor.shape[2]
    OUTPUT_DIM = y_tensor.shape[1]

    train_val_size = int(len(X_tensor) * 0.85)
    X_train_val, y_train_val = X_tensor[:train_val_size], y_tensor[:train_val_size]
    X_test, y_test = X_tensor[train_val_size:], y_tensor[train_val_size:]

    full_train_val_dataset = TensorDataset(X_train_val, y_train_val)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Final Model Training
    print("\n--- Training Final Model ---")
    final_train_loader = DataLoader(full_train_val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = BiLSTMAttentionModel(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM, dropout_prob=0.2).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    train_loss_hist, val_loss_hist = [], []

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(final_model, final_train_loader, criterion, optimizer, device)
        val_loss = evaluate_epoch(final_model, test_loader, criterion, device)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(final_model.state_dict(), MODEL_SAVE_PATH)
            print(f'   -> Model improved. Saved to "{MODEL_SAVE_PATH}"')

    # 3. Final Evaluation
    print("\n--- Final Evaluation ---")
    final_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    final_model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs, _ = final_model(inputs.to(device))
            all_preds.append(outputs.cpu().numpy())
    predictions_scaled = np.concatenate(all_preds, axis=0)

    # Inverse transform for evaluation
    y_test_numpy = y_test.numpy()
    dummy_preds = np.zeros((len(predictions_scaled), INPUT_DIM)); dummy_preds[:, :OUTPUT_DIM] = predictions_scaled
    predictions_original = scaler.inverse_transform(dummy_preds)[:, :OUTPUT_DIM]
    dummy_actuals = np.zeros_like(dummy_preds); dummy_actuals[:, :OUTPUT_DIM] = y_test_numpy
    actuals_original = scaler.inverse_transform(dummy_actuals)[:, :OUTPUT_DIM]

    # Calculate and print metrics
    mse = mean_squared_error(actuals_original, predictions_original)
    print(f"\n--- Metrics (Original Scale) ---")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    print(f"MAE:  {mean_absolute_error(actuals_original, predictions_original):.6f}")
    print(f"R²:   {r2_score(actuals_original, predictions_original):.6f}")

    print(f"\n--- Metrics (Normalized Scale) ---")
    mse_scaled = mean_squared_error(y_test_numpy, predictions_scaled)
    print(f"Scaled MSE: {mse_scaled:.6f}")
    print(f"Scaled RMSE: {np.sqrt(mse_scaled):.6f}")
    print(f"Scaled MAE:  {mean_absolute_error(y_test_numpy, predictions_scaled):.6f}")
    print(f"Scaled R²:   {r2_score(y_test_numpy, predictions_scaled):.6f}")

    # 4. Visualization
    print("\n--- Generating Plots ---")
    # Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_hist, label='Training Loss')
    plt.plot(val_loss_hist, label='Validation Loss')
    plt.title('Training and Validation Loss'); plt.xlabel('Epochs'); plt.ylabel('Loss (MSE)'); plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PLOT_SAVE_PATH, 'loss_curve.png')); plt.show()

    # Predictions vs Actuals
    feature_to_plot_idx = 1 # 'T (degC)'
    plt.figure(figsize=(15, 6))
    plt.plot(actuals_original[:200, feature_to_plot_idx], label='Actual Values')
    plt.plot(predictions_original[:200, feature_to_plot_idx], label='Predicted Values', linestyle='--')
    plt.title(f'Prediction vs Actual for {feature_names[feature_to_plot_idx]}')
    plt.xlabel('Time Steps'); plt.ylabel(feature_names[feature_to_plot_idx].split('(')[-1].replace(')',''))
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(PLOT_SAVE_PATH, 'predictions_vs_actuals.png')); plt.show()

    # Attention Visualization
    inputs, _ = next(iter(test_loader))
    sample_input = inputs[0:1].to(device)
    with torch.no_grad():
        _, attention_weights = final_model(sample_input)
    attention_weights = attention_weights.squeeze(0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(18, 2))
    im = ax.imshow(attention_weights[np.newaxis, :], cmap='Reds', aspect='auto')
    ax.set_yticks([]); ax.set_xlabel('Time Steps in Input Sequence')
    ax.set_title('Attention Weights for a Single Prediction')
    fig.colorbar(im, orientation='horizontal', pad=0.2).set_label('Attention Weight')
    plt.savefig(os.path.join(PLOT_SAVE_PATH, 'attention_visualization.png')); plt.show()

    print("\nTraining and evaluation complete.")