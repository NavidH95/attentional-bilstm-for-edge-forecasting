from periphery import GPIO
import time

# GPIOA6 = GPIO number 6
gpio_pin = 6

# Initialize GPIO6 as output
led = GPIO(gpio_pin, "out")

try:
    print("Turning LED ON")
    led.write(True)  # Turn LED ON
    time.sleep(2)

    print("Turning LED OFF")
    led.write(False)  # Turn LED OFF
    time.sleep(2)
    led.write(True)
    time.sleep(2)
    led.write(False)
    time.sleep(2)


finally:
    # Always release the GPIO when done
    led.close()
from periphery import GPIO
import time

# Export GPIO6 (PA6) and set as output
gpio = GPIO(7, "out")

# Turn LED on
gpio.write(True)
time.sleep(2)

# Turn LED off
gpio.write(False)
time.sleep(2)

# Clean up
gpio.close()

