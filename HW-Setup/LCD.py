from periphery import GPIO
import time
# Pin mapping
RS = GPIO(2, "out")   # PA2
E  = GPIO(1, "out")   # PA1
D4 = GPIO(67, "out")  # PG3 (change if mapped to PG6 = GPIO198)
D5 = GPIO(200, "out") # PG8
D6 = GPIO(199, "out") # PG7
D7 = GPIO(198, "out") # PG6

# Utility functions
def pulse_enable():
    print("PULSE E")
    E.write(True)
    time.sleep(0.005)
    E.write(False)
    time.sleep(0.005)

def write_nibble(nibble):
    D4.write(((nibble >> 0) & 1)==1)
    D5.write(((nibble >> 1) & 1)==1)
    D6.write(((nibble >> 2) & 1)==1)
    D7.write(((nibble >> 3) & 1)==1)
    pulse_enable()

def send_byte(data, is_data=True):
    RS.write(is_data)
    write_nibble((data >> 4) & 0x0f)   # Send high nibble
    write_nibble(data & 0x0F) # Send low nibble
    time.sleep(0.002)

def lcd_command(cmd):
    send_byte(cmd, is_data=False)

def lcd_write(char):
    send_byte(ord(char), is_data=True)

def lcd_init():
    time.sleep(0.05)
    RS.write(False)
    E.write(False)

    write_nibble(0x03)  # Reset
    time.sleep(0.005)

    write_nibble(0x03)  # Reset
    time.sleep(0.005)

    write_nibble(0x03)  # Reset
    time.sleep(0.005)

    write_nibble(0x02)  # 4-bit mode
    time.sleep(0.005)

    lcd_command(0x028)  # 2-line, 5x8 dots
    time.sleep(0.005)

    lcd_command(0x0C)  # Display on, cursor off
    time.sleep(0.005)

    lcd_command(0x06)
    time.sleep(0.005)

    lcd_command(0x01)
    time.sleep(0.005)

def lcd_print(msg):
    for c in msg:
        lcd_write(c)
        print(c)

# Initialize and write
lcd_init()
lcd_print("Hello, World!")
