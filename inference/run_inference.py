import numpy as np
import time
import os
import tflite_runtime.interpreter as tflite

# --- Configuration ---
TFLITE_MODEL_PATH = "outputs/saved_models/model.tflite"
SCALER_PATH = "outputs/saved_models/scaler_params.npz"

# Model Hyperparameters
SEQUENCE_LENGTH = 128
INPUT_DIM = 14
OUTPUT_DIM = 14

# --- Main Script ---
if __name__ == "__main__":
    if not os.path.exists(TFLITE_MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"Error: Model or scaler not found.")
        print("Please run `train/train.py` and `inference/convert_to_tflite.py` first.")
    else:
        # Load TFLite model and scaler
        interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()

        with np.load(SCALER_PATH) as data:
            scaler_min = data['min']
            scaler_scale = data['scale']

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Find the correct prediction output tensor
        prediction_output_index = -1
        for detail in output_details:
            if tuple(detail['shape']) == (1, OUTPUT_DIM):
                prediction_output_index = detail['index']
                break
        if prediction_output_index == -1:
            raise RuntimeError("Could not find prediction output tensor in TFLite model.")

        # Prepare a random sample input
        raw_input_data = np.random.rand(SEQUENCE_LENGTH, INPUT_DIM).astype(np.float32)
        scaled_input_data = (raw_input_data - scaler_min) * scaler_scale
        final_input_data = np.expand_dims(scaled_input_data, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], final_input_data)

        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()

        scaled_prediction = interpreter.get_tensor(prediction_output_index)

        # De-normalize the output
        final_prediction = (scaled_prediction / scaler_scale[:OUTPUT_DIM]) + scaler_min[:OUTPUT_DIM]

        print("--- TFLite Inference Test ---")
        print(f"Inference Time: {(end_time - start_time) * 1000:.4f} ms")
        print(f"Final Human-Readable Prediction: \n{final_prediction}")