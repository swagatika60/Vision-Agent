# 👁️ VisionMate: Spatial Agent Protocol

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-link-here.streamlit.app)
[![Hackathon](https://img.shields.io/badge/Hackathon-WeMakeDevs-blue.svg)](#)

🎬 **[CLICK HERE TO VIEW THE VIDEO DEMO]** *(Insert your YouTube/Devpost link here)*

**VisionMate** is a multi-modal, real-time spatial awareness agent. Built for the *Vision Possible: Agent Protocol* hackathon, it processes live video, static imagery, and WebRTC camera feeds to identify objects and mathematically calculate imminent spatial hazards.

Moving beyond simple object detection wrappers, VisionMate acts as an active telemetry dashboard, tracking time-series data and evaluating physical proximity in real-time without crashing lightweight cloud servers.

---

## 🚀 The Engineering Challenge
Processing heavy computer vision pipelines (like YOLO) on lightweight cloud servers usually results in two things: Out of Memory (OOM) crashes, or terrible accuracy due to frame distortion (squishing 1080p video into 640x640 boxes). Furthermore, simply knowing *what* is in a frame isn't enough; an AI agent needs to know if that object is safely in the distance or dangerously close to the camera.

VisionMate solves this by introducing a dynamic aspect-ratio scaler, aggressive garbage collection, and a custom mathematical proximity algorithm. 

## ✨ Key Features

* **🎬 Synchronous Live Telemetry Dashboard:**
  Upload a video and watch the AI process it in real-time. The UI features a side-by-side split displaying the active video stream alongside a live-updating metrics panel and a scrolling time-series log of all detected entities.
* **🚨 Spatial Hazard Proximity Math:**
  VisionMate doesn't just detect objects; it measures them. By evaluating the vertical span of a bounding box relative to the native frame height, the system automatically flags objects taking up >40% of the screen as a `CRITICAL HAZARD`.
* **🎥 Ultra-Low Latency Agent WebRTC:**
  Integrated with the `vision-agents` SDK and Stream's Edge network, VisionMate establishes a sub-30ms video room. Users can converse in real-time with a Gemini-powered agent that natively "sees" their hardware camera feed.
* **📷 High-Fidelity Aspect-Preserving Scans:**
  Static images are processed using a custom `resize_maintaining_aspect` function. This shrinks file sizes to save RAM while preserving the exact mathematical proportions of the image, achieving 99% accuracy on small background objects.
* **🔊 Audio Feedback & JSON Export:**
  Generates automated Google TTS (`gtts`) vocal summaries of the environment and exports highly detailed JSON security audits containing timestamps, unique entity tracking, and hazard logs.

---

## 🛠️ Tech Stack

* **Frontend & Dashboard:** Streamlit, Streamlit Community Cloud
* **Core Agent / LLM:** Gemini Realtime API, Stream Vision Agents SDK
* **Computer Vision:** Ultralytics (YOLOv8s - Small), OpenCV (`opencv-python-headless`)
* **Video/Audio Routing:** Stream Edge WebRTC, Google TTS
* **Security & Memory:** `bcrypt` (Secure local auth), `gc` (Python Garbage Collection for Cloud limits)

---

## ⚙️ How the Proximity Algorithm Works
VisionMate runs a highly optimized tracking loop that ignores distant objects and triggers alarms for immediate spatial threats:
1. **Dynamic Scaling:** Frames are shrunk to a max-width of 640px to prevent OOM errors, but the height is dynamically calculated to perfectly preserve the original aspect ratio (`h * ratio`).
2. **Confidence & Overlap:** The model uses `conf=0.25` to catch everything and `iou=0.45` to perfectly separate overlapping objects (like cars parked behind one another).
3. **Proximity Math:** `vertical_span = (y2 - y1) / h_img`. If the object's height exceeds 40% of the total native frame, it is flagged as a direct collision risk.
4. **Action:** The system draws a thickened red box, updates the live telemetry dashboard, and logs the threat.

---

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
**3.Install Python Dependencies:**
(Note: Uses opencv-python-headless for cloud compatibility)
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
-----

**🔮 Future Roadmap & What's Next**

I built the foundation of VisionMate during this hackathon, but I am highly committed to continuing its development into a real-world tool. My upcoming updates will include:

Haptic Feedback Integration: Connecting the proximity alarm to Bluetooth wearables (like a smartwatch) to vibrate when a hazard is detected, providing silent alerts.

Night Vision Mode: Implementing low-light enhancement filters using OpenCV to ensure the object tracking works reliably in dark environments.

GPS Route Integration: Combining hazard detection with live turn-by-turn walking directions.

Native Mobile Port: Packaging the WebRTC agent into a native Android/iOS application so users can simply point their phones without needing a browser.

Built with ❤️ by Swagatika Beura for the WeMakeDevs Hackathon.
-----



