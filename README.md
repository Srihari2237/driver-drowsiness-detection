# 🚗 Driver Drowsiness Detection System

## 📌 Overview

The **Driver Drowsiness Detection System** is a real-time computer vision application that monitors a driver’s alertness using facial landmarks. It detects signs of fatigue such as eye closure and yawning, and alerts the driver to prevent accidents.

---

## 🎯 Features

* 👁️ Eye Aspect Ratio (EAR) based drowsiness detection
* 😮 Mouth Aspect Ratio (MAR) based yawning detection
* 🚨 Real-time alert system (alarm sound)
* 📊 Live EAR trend graph
* 📉 Fatigue score monitoring
* 🎨 Modern dashboard UI using Streamlit
* 🎥 Real-time webcam integration

---

## 🧠 Technologies Used

* **Python**
* **OpenCV** – Video processing
* **MediaPipe** – Facial landmark detection
* **NumPy & SciPy** – Mathematical computations
* **Streamlit** – Web-based UI
* **PIL (Pillow)** – Image processing

---

## ⚙️ How It Works

1. Captures live video from webcam
2. Detects face and extracts facial landmarks
3. Computes:

   * **EAR (Eye Aspect Ratio)** → Detects eye closure
   * **MAR (Mouth Aspect Ratio)** → Detects yawning
4. Classifies driver state:

   * ALERT
   * SEMI-DROWSY
   * DROWSY
   * YAWNING
5. Triggers alarm if fatigue is detected
6. Displays results in a real-time dashboard

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📊 Output

* Real-time video feed
* Driver status display (ALERT / DROWSY / YAWNING)
* EAR & MAR values
* Fatigue score indicator
* Live graph of eye activity

---

## 🖼️ Screenshots

> (Add your screenshots here for better presentation)

Example:

* Dashboard UI
* Alert state
* Graph visualization

---

## 📁 Project Structure

```
driver-drowsiness-detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── detector.py
│   ├── ear.py
│   ├── mar.py
│   └── alarm.py
│
├── utils/
│   ├── config.py
│   └── visualization.py
│
├── assets/
│   └── sounds/
│
└── .gitignore
```

---

## 🧠 Applications

* 🚗 Driver safety systems
* 🚛 Fleet monitoring
* 🚘 Smart vehicles
* 🏥 Fatigue detection in workplaces

---

## ⚠️ Limitations

* Depends on lighting conditions
* Requires visible face
* Webcam-based system (local execution preferred)

---

## 🔮 Future Enhancements

* Voice alert system
* Mobile app integration
* Cloud deployment with WebRTC
* Deep learning model integration

---

## 👨‍💻 Author

**Srihari**

---

## ⭐ Acknowledgement

This project demonstrates the application of **Machine Learning and Computer Vision** techniques in real-world safety systems.

---

## 📌 Note

For best performance, run the application locally with webcam support.

---
