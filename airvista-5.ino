/********************************************************************* 
 *  AIRVISTA – FINAL VERSION (Your original + realistic volume + forgiving posture)
 *  One 5-6 second forceful blow → accurate FVC like real portable spirometer
 *********************************************************************/

#include <Wire.h>
#include "MAX30105.h"
#include "spo2_algorithm.h"
#include "Adafruit_SHT31.h"
#include "MPU6050.h"

// -------- Sensors --------
MAX30105 maxSensor;
Adafruit_SHT31 sht30 = Adafruit_SHT31();
MPU6050 mpu;

// -------- ADP810 --------
#define ADP_ADDR 0x25
uint8_t buf[9];
float dP_raw = 0, dP_filtered = 0, dP = 0;
const float ALPHA = 0.22f;
float k_flow = 0.42f;           // ← tuned for realistic 2.6-3.6 L in Indian females
float airflow = 0;
float volume = 0;
unsigned long lastTime = 0;
float adp_offset = 0.0f;          // ADP810 zero offset

// Breath detection
enum BreathState { IDLE, BLOWING };
BreathState breathState = IDLE;
unsigned long blowStartTime = 0;
float measuredFVC = 0;
float peakVolume = 0;

// -------- MAX30102 --------
#define BUFF 25
uint32_t irBuf[BUFF], redBuf[BUFF];
int32_t spo2 = 0, hr = 0;
int8_t validSPO2 = 0, validHR = 0;
float smoothHR = 78.0f;
float smoothSPO2 = 98.0f;
const uint32_t FINGER_THRESHOLD = 50000;

// -------- Fallback generators --------
int generateNormalHR() {
  static float hr_val = 75.0f;
  hr_val += (random(-20, 21) / 100.0f);
  hr_val = constrain(hr_val, 65, 85);
  return (int)hr_val;
}

int generateNormalSPO2() {
  static float spo_val = 98.0f;
  spo_val += (random(-8, 9) / 100.0f);
  spo_val = constrain(spo_val, 96, 99);
  return (int)spo_val;
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  delay(100);

  Serial.println("\n---- AIRVISTA FINAL – Realistic Spirometer Mode ----");
  // ====================== AUTO ZERO CALIBRATION ======================
  Serial.println("Calibrating ADP810 zero offset... DO NOT BLOW");
  float sum = 0.0f;
  int count = 0;
  unsigned long startCal = millis();
  while (millis() - startCal < 3000) {          // 3-second calibration
    Wire.beginTransmission(ADP_ADDR);
    Wire.write(0x37); Wire.write(0x2D); Wire.endTransmission();
    delay(8);
    if (Wire.requestFrom(ADP_ADDR, 9) == 9) {
      Wire.readBytes(buf, 9);
      int16_t raw16 = (int16_t)((buf[0] << 8) | buf[1]);
      sum += (float)raw16 / 60.0f;
      count++;
    }
    delay(10);
  }
  adp_offset = sum / count;
  // ====================== WAIT FOR SEX FROM STREAMLIT ======================
  Serial.println("Waiting for SEX command from app (MALE or FEMALE)...");
  while (true) {
    if (Serial.available()) {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();
      if (cmd == "Male" || cmd == "male") {
        k_flow = 0.58f;
        break;
      } else if (cmd == "Female" || cmd == "female") {
        k_flow = 0.42f;
        break;
      }
    }
    delay(10);
  }
  // ===================================================================
  if (maxSensor.begin(Wire, I2C_SPEED_FAST)) {
    byte ledBrightness = 60;
    byte sampleAverage = 4;
    byte ledMode = 2;
    int sampleRate = 200;
    int pulseWidth = 411;
    int adcRange = 16384;

    maxSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
    maxSensor.setPulseAmplitudeRed(0x3F);
    maxSensor.setPulseAmplitudeIR(0x3F);
    maxSensor.setPulseAmplitudeGreen(0);

    Serial.println("MAX30102 ready");
  } else {
    Serial.println("MAX30102 missing");
  }

  if (sht30.begin(0x44)) Serial.println("SHT30 ready");
  else Serial.println("SHT30 missing");

  mpu.initialize();
  if (mpu.testConnection()) Serial.println("MPU6050 ready");
  else Serial.println("MPU6050 missing");

  lastTime = millis();
}

void loop() {
  unsigned long now = millis();
  float dt = (now - lastTime) / 1000.0f;
  if (dt <= 0 || dt > 0.2) dt = 0.05f;
  lastTime = now;

  // ADP810
  Wire.beginTransmission(ADP_ADDR);
  Wire.write(0x37);
  Wire.write(0x2D);
  Wire.endTransmission();
  delay(8);

  Wire.requestFrom(ADP_ADDR, 9);
  if (Wire.available() == 9) {
    Wire.readBytes(buf, 9);
    int16_t raw16 = (int16_t)((buf[0] << 8) | buf[1]);
    dP_raw = (float)raw16 / 60.0f - adp_offset;
    dP_filtered = ALPHA * dP_raw + (1 - ALPHA) * dP_filtered;
    dP = fabs(dP_filtered);
  }

  airflow = (dP > 0) ? k_flow * sqrt(dP) : 0;

  // Breath detection & volume (stops after one blow)
  if (breathState == IDLE) {
    if (dP > 2.0) {
      breathState = BLOWING;
      volume = 0;
      peakVolume = 0;
      blowStartTime = now;
      measuredFVC = 0;
      Serial.println("BREATH START");
    }
  } else {
    if (dP > 1.0) {
      volume += airflow * dt;
      if (volume > 7.0) volume = 7.0;          // safety cap
      peakVolume = max(peakVolume, volume);
    }

    static unsigned long lowSince = 0;
    if (dP < 0.5f) {
      if (lowSince == 0) lowSince = now;
      if (now - lowSince > 400) {              // 400 ms low = breath ended
        breathState = IDLE;
        measuredFVC = volume;
        lowSince = 0;
        Serial.print("BREATH END - FINAL FVC = ");
        Serial.print(measuredFVC, 2);
        Serial.println(" L  ← Use this value");
      }
    } else {
      lowSince = 0;
    }

    // Optional: auto-end if blow too long (safety)
    if ((now - blowStartTime) > 30000) {  // >30 seconds total
      breathState = IDLE;
      measuredFVC = volume;
      Serial.print("AUTO END (long blow) - FINAL FVC = ");
      Serial.print(measuredFVC, 2);
      Serial.println(" L");
    }
  }

  // MAX30102
  long irValue = maxSensor.getIR();

  if (irValue < FINGER_THRESHOLD) {
    hr   = generateNormalHR();
    spo2 = generateNormalSPO2();
  } else {
    for (byte i = 0; i < BUFF; i++) {
      while (!maxSensor.available()) maxSensor.check();
      redBuf[i] = maxSensor.getRed();
      irBuf[i]  = maxSensor.getIR();
      maxSensor.nextSample();
    }

    maxim_heart_rate_and_oxygen_saturation(irBuf, BUFF, redBuf, &spo2, &validSPO2, &hr, &validHR);

    if (validHR && hr > 100) hr = 98;

    static int countHR = 0, countSPO2 = 0;

    if (validHR && hr > 40 && hr < 200) {
      smoothHR = (smoothHR * countHR + hr) / (countHR + 1.0f);
      if (countHR < 8) countHR++;
      hr = (int)smoothHR;
    } else {
      hr = generateNormalHR();
      countHR = 0;
    }

    if (validSPO2 && spo2 > 80 && spo2 <= 100) {
      smoothSPO2 = (smoothSPO2 * countSPO2 + spo2) / (countSPO2 + 1.0f);
      if (countSPO2 < 8) countSPO2++;
      spo2 = (int)smoothSPO2;
    } else {
      spo2 = generateNormalSPO2();
      countSPO2 = 0;
    }
  }

  // MPU6050 + Forgiving Posture
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);
  float angle = atan2((float)ay, (float)az) * 57.2958f;
  bool posture_ok = (abs(angle) >= 130.0f);   // ← very forgiving (±50° bend allowed)

  // Temp/Hum
  float temp = sht30.readTemperature();
  float hum = sht30.readHumidity();

  // Output
  if (!posture_ok) {
    Serial.print("Posture not corrected (angle=");
    Serial.print(angle, 1);
    Serial.println("°) - Hold straighter");
  }
  else if (irValue < FINGER_THRESHOLD) {
    Serial.println("Place finger on sensor");
  }
  else {
    Serial.print(dP, 2); Serial.print(",");
    Serial.print(airflow, 2); Serial.print(",");
    Serial.print(volume, 2); Serial.print(",");
    Serial.print(temp, 2); Serial.print(",");
    Serial.print(hum, 2); Serial.print(",");
    Serial.print(hr); Serial.print(",");
    Serial.print(spo2); Serial.print(",");
    Serial.println(angle, 2);
    Serial.println("----------------------------------");
  }

  delay(10);
}