void setup() {
  Serial.begin(9600);
}

void loop() {
  int gasValue = analogRead(A0); // 0–1023
  Serial.println(gasValue);
  delay(1000);
}
