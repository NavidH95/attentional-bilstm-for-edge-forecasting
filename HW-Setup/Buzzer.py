from periphery import GPIO
import time

# GPIO6 is PA6
buzzer = GPIO(6, "out")

# Turn ON (set LOW)
buzzer.write(False)
time.sleep(1)

# Turn OFF (set HIGH)
buzzer.write(True)
time.sleep(1)

buzzer.close()
