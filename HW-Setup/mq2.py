#sudo apt install python3-serial 
import serial

ser = serial.Serial('/dev/ttyS1', 9600)  # Adjust for your UART
while True:
    line = ser.readline().decode().strip()
    print(f"Gas value: {line}")
