# 🥊 Gesture-Based Boxing Game Controller

Control a boxing game using real-time hand gestures through your webcam! This project uses **OpenCV**, **MediaPipe**, and **PyAutoGUI** to detect your gestures and simulate key presses in the browser to control the game [Big Shot Boxing](https://poki.com/en/g/big-shot-boxing).

---

## 🎯 Features

- 👊 Detects **Left Punch**, **Right Punch**, **Uppercut**, and **Block** gestures.
- 🖥️ Launches the boxing game in your browser automatically.
- 🧠 Uses **MediaPipe** for accurate hand tracking.
- ⌨️ Uses **PyAutoGUI** to simulate game key presses.
- 🎮 Real-time visual feedback via OpenCV.
- 🚀 Includes gesture cooldown and movement threshold to prevent accidental triggers.

---

## 🤖 How It Works

1. **MediaPipe** identifies wrist positions from both hands.
2. Movement direction and speed are calculated frame-to-frame.
3. Based on the position and velocity:
   - Left Punch: Left hand quickly moves right.
   - Right Punch: Right hand quickly moves left.
   - Uppercut: One hand raised above the face.
   - Block: Both hands close to the face.
4. Detected gesture triggers a simulated keypress using **PyAutoGUI**.

---

## 🎮 Gesture Mapping

| Gesture       | Key Press | Detection Logic                          |
|---------------|-----------|------------------------------------------|
| Left Punch    | `←`       | Left hand moves rapidly to the right     |
| Right Punch   | `→`       | Right hand moves rapidly to the left     |
| Uppercut      | `X`       | One hand raised above the face           |
| Block         | `Z`       | Both hands close to the face area        |

---

## 📦 Installation

Make sure you have **Python 3.7+** installed, then run:

```bash
pip install opencv-python mediapipe pyautogui numpy
