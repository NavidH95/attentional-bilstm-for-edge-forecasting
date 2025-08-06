from periphery import GPIO
import time
import tflite_runtime.interpreter as tflite
import numpy as np

RS = GPIO(2, "out")
E  = GPIO(1, "out")
D4 = GPIO(67, "out")
D5 = GPIO(200, "out")
D6 = GPIO(199, "out")
D7 = GPIO(198, "out")

def pulse_enable():
   # print("pulse E")
    E.write(True)
    time.sleep(0.005)
    E.write(False)
    time.sleep(0.005)

def write_nibble(nibble):
    D4.write(((nibble >> 0) & 1) == 1)
    D5.write(((nibble >> 1) & 1) == 1)
    D6.write(((nibble >> 2) & 1) == 1)
    D7.write(((nibble >> 3) & 1) == 1)
    pulse_enable()

def send_byte(data, is_data=True):
    RS.write(is_data)
    write_nibble((data >> 4) & 0x0f)
    write_nibble(data & 0x0f)
    time.sleep(0.002)

def lcd_command(cmd):
    send_byte(cmd, is_data=False)

def lcd_write(char):
    send_byte(ord(char), is_data=True)

def lcd_init():
    time.sleep(0.05)
    RS.write(False)
    E.write(False)

    write_nibble(0x03)
    time.sleep(0.005)

    write_nibble(0x03)
    time.sleep(0.005)

    write_nibble(0x03)
    time.sleep(0.005)

    write_nibble(0x02)
    time.sleep(0.005)

    lcd_command(0x028)
    time.sleep(0.005)

    lcd_command(0x0C)
    time.sleep(0.005)

    lcd_command(0x06)
    time.sleep(0.005)

    lcd_command(0x01)
    time.sleep(0.005)

def lcd_print(msg):
    for c in msg:
        lcd_write(c)
    #    print(c)

def testperdict():
    # --- Step 1: Load Model and Scaler Parameters ---
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()

    with np.load('scaler_params.npz') as data:
        scaler_min = data['min']
        scaler_scale = data['scale']

    print("Model and scaler parameters loaded successfully.")

    # --- Step 2: Get Model Input and Output Details ---
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    SEQUENCE_LENGTH = input_shape[1] # Should be 128
    INPUT_DIM = input_shape[2]       # Should be 14

    # --- NEW: Identify the correct output tensor ---
    # The TFLite model has multiple outputs. We need to find the one for predictions,
    # which has the shape (1, 14), NOT the attention weights (1, 128).
    prediction_output_index = -1
    print("\n--- Available Model Outputs ---")
    for i, details in enumerate(output_details):
        print(f"Output {i}: shape={details['shape']}")
        # The correct output has INPUT_DIM (14) features
        if tuple(details['shape']) == (1, INPUT_DIM):
            prediction_output_index = details['index']
            print(f"-> Found prediction tensor at index {i}.")

    if prediction_output_index == -1:
        raise RuntimeError(f"Could not find an output tensor with shape (1, {INPUT_DIM}). Please check the exported model.")
    print("---------------------------\n")


    # --- Step 3: Prepare a Sample Input ---
    raw_input_data = np.random.rand(SEQUENCE_LENGTH, INPUT_DIM)


    # --- Step 4: Normalize the Input Data (Manually) ---
    scaled_input_data = (raw_input_data - scaler_min) * scaler_scale
    final_input_data = np.expand_dims(scaled_input_data, axis=0).astype(np.float32)


    # --- Step 5: Run Inference ---
    interpreter.set_tensor(input_details[0]['index'], final_input_data)
    interpreter.invoke()

    # Get the result from the CORRECT output tensor using the identified index
    scaled_prediction = interpreter.get_tensor(prediction_output_index)

    print(f"\nShape of the selected prediction tensor: {scaled_prediction.shape}") # Should be (1, 14) now


    # --- Step 6: De-normalize the Output (Manually) ---
    # Now this operation should work perfectly because shapes match: (1, 14) and (14,)
    final_prediction = (scaled_prediction / scaler_scale) + scaler_min

    print(f"\nFinal Human-Readable Prediction: \n{final_prediction}")
    first_prediction = f"{final_prediction[0][0]:.2f}"
    second_prediction = f"{final_prediction[0][1]:.2f}"
    print("first prediction" + first_prediction + "second prediction" + second_prediction)
    lcd_print("P1:"+first_prediction)
    lcd_command(0xC0)
    time.sleep(0.005)
    lcd_print("P2:"+second_prediction)

lcd_init()
testperdict()
print("end")
