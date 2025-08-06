#sudo apt install python3-serial
import serial

ser = serial.Serial('/dev/ttyS1', 9600)  # Adjust based on your UART port

while True:
    line = ser.readline().decode().strip()
    print(f"🌱 Moisture: {line}")
