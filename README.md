# 👁️ VisionMate: Spatial Agent Protocol

<p align="center">
  <img src="https://dev-to-uploads.s3.amazonaws.com/uploads/articles/b82pyn7by1b3jwcn0ifl.jpeg" alt="VisionMate Banner" width="800">
</p>

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vision-agent.streamlit.app/)
[![Hackathon](https://img.shields.io/badge/Hackathon-WeMakeDevs-blue.svg)](https://www.wemakedevs.org/hackathons/vision)
[![Vision Agents](https://img.shields.io/badge/Powered%20By-Vision%20Agents-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

{% youtube Y04gNrPA3kk %}
🎬 **[WATCH THE VIDEO DEMO](https://youtube.com/watch?v=Y04gNrPA3kk)** **VisionMate AI** is a multi-modal, real-time spatial awareness agent. Developed for the *Vision Possible: Agent Protocol* hackathon, it transforms standard camera feeds into **active telemetry dashboards** that mathematically calculate imminent spatial hazards.

Unlike passive object detectors, VisionMate evaluates physical proximity in real-time to provide life-saving alerts—all while optimized for lightweight cloud infrastructure (1GB RAM).

---

## 🚀 The Engineering Challenge:

Processing heavy computer vision (YOLO) on cloud servers typically results in **Out of Memory (OOM) crashes** or distorted detection accuracy. Furthermore, an AI agent needs to distinguish between a car that is "visible" and a car that is "imminent danger."

**VisionMate solves this with:**
1.  **Dynamic Aspect-Ratio Scaling:** Prevents frame "squishing" while reducing RAM footprint.
2.  **Edge Network Routing:** Utilizing **Stream's Vision Agents SDK** for sub-30ms latency.
3.  **The 40% Rule:** A custom proximity algorithm based on vertical frame occupancy.

---

## ✨ Key Features

* **🎬 Synchronous Telemetry Dashboard:** View side-by-side live processing with scrolling time-series logs and real-time entity tracking.
* **🚨 Spatial Hazard Logic:** Automatically flags objects taking up **>40% of the screen** as a `CRITICAL HAZARD` using custom geometric math.
* **🎥 Ultra-Low Latency WebRTC Agent:** Uses the **Stream Vision Agents SDK** to bridge local hardware cameras with **Gemini Realtime API** for instant voice-to-vision interaction.
* **📷 High-Fidelity Scans:** A custom `resize_maintaining_aspect` function ensures accuracy even on low-memory cloud instances.
* **🔊 Audio Alerts & JSON Audits:** Generates vocal warnings via `gTTS` and exports comprehensive security logs for post-incident review.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend** | Streamlit + Custom CSS |
| **AI Agent Logic** | **Stream Vision Agents SDK** |
| **Reasoning Engine** | **Google Gemini Realtime API** |
| **Computer Vision** | Ultralytics YOLOv8s (Optimized) |
| **Networking** | WebRTC via Stream Edge Network |
| **Utilities** | OpenCV, gTTS, Bcrypt, Dotenv |

---

## ⚙️ How the Proximity Algorithm Works

VisionMate runs an optimized tracking loop that calculates spatial threat using the vertical span of the object relative to the frame height.



**The Calculation:**
$$vertical\_span = \frac{y2 - y1}{h\_img}$$

* **Hazard (>0.4):** Imminent collision risk. Triggers voice alert and red HUD.
* **Warning (>0.2):** Proximity warning. Yellow HUD.
* **Safe (<0.2):** Distant object. Green HUD.

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
**. Haptic Feedback:** Integration with smartwatches for vibration-based hazard alerts.

**. Night Vision:** Low-light enhancement filters using OpenCV.

**. Native Mobile App:** Packaging the WebRTC protocol into a dedicated Android/iOS safe-navigation app.

----
Built with ❤️ by Swagatika Beura for the WeMakeDevs Hackathon.
-----



