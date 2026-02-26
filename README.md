# 👁️ Vision-Agent (Project: VisionMate)
**A real-time assistive spatial awareness system for the visually impaired.**

VisionMate is an AI-powered mobility agent designed to provide independence through spatial intelligence. By leveraging **Stream's Ultra-Low Latency Edge Network**, **Google Gemini**, and **Ultralytics**, it not only identifies surroundings but calculates depth and trajectory to provide instant, life-saving audio feedback.

## 🚀 Key Features
- **Spatial Hazard Detection**: Calculates depth to prioritize immediate dangers (e.g., "Car on your LEFT is CLOSE") before reading general scene descriptions.
- **Unique Object Tracking**: Uses persistent ID tracking to prevent over-counting moving objects, ensuring the user gets "Sure Numbers" (e.g., exactly 2 cars, not 14 detections).
- **Real-Time WebRTC Agent**: Processes environment data in under 30ms using the Stream Edge network.
- **Multi-Modal AI**: Combines Gemini for high-level conversation, Moondream for visual QA, and YOLOv8 for rapid obstacle tracking.
- **Assistive Voice Feedback**: Conversational AI that interrupts itself to warn the user of immediate physical obstacles.

## 🛠️ Tech Stack
- **SDK**: `vision-agents` (Stream Edge Network)
- **LLMs & Vision**: Google Gemini, Ultralytics YOLOv8, Moondream, BLIP
- **Frontend**: Streamlit
- **Language**: Python (v3.10+)

## 🔧 Installation & Setup (Zero-Error Guide)

To run Vision-Agent locally or on a fresh GitHub Codespace without any `libGL` system errors, please run this exact sequence of commands in your terminal:

```bash
# 1. Clone the repository and enter the project directory
git clone [https://github.com/swagatika60/Vision-Agent.git](https://github.com/swagatika60/Vision-Agent.git)
cd Vision-Agent

# 2. Install required Linux system graphics drivers 
# (This permanently fixes the libGL.so.1 OpenCV error on cloud servers/Codespaces)
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0

# 3. Create a fresh Python virtual environment
python3 -m venv venv

# 4. Activate the virtual environment
# On macOS/Linux/GitHub Codespaces:
source venv/bin/activate
# On Windows (if running locally):
# venv\Scripts\activate

# 5. Upgrade pip to prevent any caching or installation bugs
python -m pip install --upgrade pip

# 6. Install all locked project dependencies from requirements.txt
pip install -r requirements.txt

# 7. The "Bulletproof" Override: 
# Ensures the server-safe version of OpenCV is used, removing hidden conflicts.
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-python-headless --force-reinstall

# 8. Run the application
streamlit run visionmate.py

---

