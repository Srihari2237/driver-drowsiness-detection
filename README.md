# 🚗 Driver AI Monitoring System

## 📌 Overview

The **Driver AI Monitoring System** is an advanced real-time computer vision application designed to enhance road safety by detecting driver fatigue and distraction.
It uses facial landmarks, eye tracking, head pose estimation, and behavioral analysis to monitor the driver's alertness continuously.

---

## 🎯 Key Features

* 👁️ **Drowsiness Detection (EAR)**
* 😮 **Yawning Detection (MAR)**
* 👀 **Gaze Tracking (Pupil Movement)**
* 🧠 **Head Pose Detection (Yaw & Drop)**
* 🚨 **Real-Time Alarm System**
* 📊 **Live Analytics Dashboard (Streamlit UI)**
* 📈 **Multi-Graph Visualization (EAR, MAR, Gaze, Head)**
* 📄 **Session Report Generation (CSV Download)**
* 🧪 **Dynamic Calibration System (Adaptive Thresholds)**
* 🕶️ **Robust to Glasses / Lighting Conditions**

---

## 🧠 Technologies Used

* **Python**
* **OpenCV** – Real-time video processing
* **MediaPipe** – Facial landmark detection
* **NumPy / Math** – Feature calculations
* **Streamlit** – Interactive dashboard UI
* **Pygame** – Alarm sound system
* **Pandas** – Data analysis & reporting

---

## ⚙️ How It Works

1. Captures live video from webcam
2. Detects facial landmarks using MediaPipe
3. Computes key metrics:

   * **EAR (Eye Aspect Ratio)** → Eye closure detection
   * **MAR (Mouth Aspect Ratio)** → Yawning detection
   * **Gaze Ratio** → Pupil direction tracking
   * **Head Pose** → Distraction & head drop
4. Applies:

   * **Smoothing filters**
   * **Dynamic calibration (first few frames)**
5. Classifies driver state:

   * ALERT
   * DROWSY
   * YAWNING
   * DISTRACTED
   * HEAD DROP
   * CRITICAL DISTRACTION
6. Triggers alarm if unsafe condition detected
7. Displays live dashboard + analytics

---

## 📊 System Architecture

```text
Webcam Input
     ↓
Face Detection (MediaPipe)
     ↓
Feature Extraction
(EAR, MAR, Gaze, Head Pose)
     ↓
Temporal Analysis (Counters + Buffers)
     ↓
State Classification
     ↓
Alarm System + UI Dashboard
```

---

## 📁 Project Structure

```
driver-drowsiness-detection/
│
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── detector.py         # Core detection logic
│   ├── ear.py              # EAR calculation
│   ├── mar.py              # MAR calculation
│   └── alarm.py            # Alarm system
│
├── utils/
│   ├── config.py           # Thresholds & settings
│   └── visualization.py    # Drawing utilities
│
├── assets/
│   └── sounds/
│       └── alarm.wav
```

---

## 🚀 Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run Application

```bash
streamlit run app.py
```

---

## 📊 Output

* 🎥 Live camera feed
* 📈 Real-time graphs (EAR, MAR, Gaze, Head)
* 🚦 Driver status indicator
* 🔊 Alarm alerts
* 📄 Downloadable session report

---

## 🧠 Detection Logic Highlights

* **Adaptive thresholds** using calibration phase
* **Smoothing buffers** to reduce noise (especially with glasses)
* **Multi-signal fusion** (eyes + mouth + head + gaze)
* **Time-based detection** using frame counters

---

## ⚠️ Limitations

* Requires good lighting conditions
* Performance depends on system hardware
* Streamlit is not a true real-time engine

---

## 🔮 Future Enhancements

* 📱 Mobile app integration
* 🌐 WebRTC deployment
* 🧠 Deep learning-based classification
* 🎤 Voice alert system
* ☁️ Cloud-based monitoring

---

## 👨‍💻 Author

**Srihari**

---

## ⭐ Project Value

This project demonstrates a **real-world AI safety system** combining:

* Computer Vision
* Human behavior analysis
* Real-time UI systems

---

## 📌 Note

For best performance, run the application on a system with good processing capability and keep the device plugged in.

---
