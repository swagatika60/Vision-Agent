# 👁️ VisionMate: The Elite Mobility Assistant 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://visionmate-agent.streamlit.app)
[![Hackathon](https://img.shields.io/badge/Hackathon-WeMakeDevs-blue.svg)](#)

**VisionMate** is a multi-modal, ultra-low latency AI mobility assistant designed to give visually impaired users real-time environmental awareness and critical hazard warnings. 

Moving beyond basic image captioning, VisionMate acts as a "digital guide dog," utilizing state-of-the-art Agent Protocols and spatial tracking to calculate imminent collision risks in real-time.

---

## 🚀 The Problem We Solved
For visually impaired individuals, knowing that there are "3 cars and 2 people" in a scene is not actionable information. They need to know if a specific car or person is **approaching too closely**. VisionMate solves this by tracking objects and mathematically calculating their proximity to the user, filtering out background noise to deliver only the most critical safety alerts.

## ✨ Key Features

* **🚨 Spatial Hazard Tracking (Video Analysis):**
  Uses YOLOv8 object tracking combined with bounding-box verticality analysis. If an object (like a car or bike) takes up more than 40% of the camera frame, VisionMate isolates it as a `HAZARD`, drawing a red bounding box and preparing an urgent voice alert.
* **🎥 Ultra-Low Latency Agent Protocol (Live Call):**
  Integrates the `vision-agents` SDK and Stream Edge network to establish a sub-30ms WebRTC video room. Users can converse in real-time with a Gemini Realtime LLM that can literally "see" through their camera.
* **🖼️ Depth-Aware Scene Analysis:**
  Combines Salesforce's BLIP image captioning with YOLO object detection to provide rich, descriptive context of static environments.
* **🔊 Accessible Audio Feedback:**
  Fully integrated Text-to-Speech (TTS) ensures all hazard reports, safe path confirmations, and AI insights are spoken aloud to the user automatically.

---

## 🛠️ Tech Stack

* **Frontend & Deployment:** Streamlit, Streamlit Community Cloud
* **Core AI / LLM:** Gemini Realtime API, Vision Agents SDK
* **Computer Vision:** Ultralytics (YOLOv8 Nano), OpenCV (`opencv-python-headless`)
* **Image Processing:** HuggingFace Transformers (BLIP), Pillow
* **Video/Audio Routing:** Stream Edge WebRTC, Google TTS (`gtts`)
* **Security:** `bcrypt` (Secure local authentication), `python-dotenv`

---

## ⚙️ How the Proximity Algorithm Works
VisionMate runs a highly optimized tracking loop that ignores distant, safe objects and triggers alarms for close objects.
1. **Filtering:** The model uses `conf=0.5` and a strict class filter `[0, 1, 2, 3, 5, 7]` to only look for real obstacles (People, Bicycles, Cars, Motorcycles, Buses).
2. **Proximity Math:** `box_height = (y2 - y1) / h_img`. If the object's height exceeds 40% of the total frame, it is flagged as a direct collision risk.
3. **Action:** The system generates an exportable JSON hazard report and triggers a critical audio warning.

---

## 💻 Local Installation & Setup

If you wish to run VisionMate locally or in a Codespace, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/swagatika60/Vision-Agent.git](https://github.com/swagatika60/Vision-Agent.git)
cd Vision-Agent
```
**2.Install System Video Drivers (Linux/Codespaces only):**
```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```
**Install Python Dependencies:**
```bash
pip install -r requirements.txt
```
**4.Set up Environment Variables:**
Create a .env file in the root directory and add your keys:
```bash
GEMINI_API_KEY="your_gemini_key"
STREAM_API_KEY="your_stream_key"
STREAM_API_SECRET="your_stream_secret"
```
**5. Run the Application:**
```bash
python -m streamlit run visionmate.py
```
**6. Access the App via Port 8501:**

Running on your Local Computer: Open your web browser and go to http://localhost:8501

**🔮 Future Roadmap & What's Next**
I built the foundation of VisionMate during this hackathon, but I am highly committed to continuing its development into a real-world tool. My upcoming updates will include:

Haptic Feedback Integration: Connecting the proximity alarm to Bluetooth wearables (like a smartwatch) to vibrate when a hazard is detected, providing silent alerts.

Night Vision Mode: Implementing low-light enhancement filters using OpenCV to ensure the object tracking works reliably in dark environments.

GPS Route Integration: Combining hazard detection with live turn-by-turn walking directions.

Native Mobile Port: Packaging the WebRTC agent into a native Android/iOS application so users can simply point their phones without needing a browser.

Built with ❤️ by Swagatika Beura for the WeMakeDevs Hackathon.
-----



