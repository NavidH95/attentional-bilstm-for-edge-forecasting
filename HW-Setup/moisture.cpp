void setup() {
  Serial.begin(9600);
}

void loop() {
  int moisture = analogRead(A0); // Range 0–1023
  Serial.println(moisture);
  delay(1000);
}
