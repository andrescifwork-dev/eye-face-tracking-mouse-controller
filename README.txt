# AI-Based Face Tracking Mouse Controller – Raspberry Pi + ESP32

This project implements a functional human-computer interface using a Raspberry Pi 5, Python, OpenCV, and TensorFlow. It uses a webcam to detect a person's masked face and estimate gaze direction to control the mouse cursor on any device. Additional facial gestures such as eye closure and mouth opening are used to trigger clicks and switch between control modes (cursor vs scroll). The system sends commands via Bluetooth to an ESP32, which acts as a mouse controller for the target device.

## 🧠 Key Features

- 🎯 Real-time face tracking and gaze estimation
- 😷 Masked face detection using TensorFlow
- 👁️ Eye closure detection for click simulation
- 👄 Mouth gesture detection to switch modes
- 🖱️ Cursor and scroll control via facial gestures
- 📡 Bluetooth communication with ESP32 for device control

## 🔧 Technologies and Tools

- Raspberry Pi 5
- Python 3, OpenCV (`cv2`), TensorFlow
- Webcam (USB)
- ESP32 microcontroller
- Bluetooth HID protocol
- Custom-trained mask face detection model

## 📁 Project Structure

- `src/`: Python scripts for image processing and Bluetooth communication
- `models/`: TensorFlow model for masked face detection
- `img/`: System diagrams and screenshots
- `docs/`: Technical documentation and flowcharts
- `esp32/`: Arduino code for mouse control via Bluetooth

## 🎥 Demo

[Watch demo video](https://www.youtube.com/watch?v=demo-link) *(replace with actual link)*

## 🚀 How It Works

1. Raspberry Pi captures webcam feed and processes it using OpenCV and TensorFlow.
2. Gaze direction is mapped to cursor movement.
3. Eye closure triggers a mouse click.
4. Mouth gesture toggles between cursor and scroll mode.
5. Commands are sent via Bluetooth to ESP32, which emulates mouse actions on the target device.

## 📄 License

This project is released under the MIT License.

---

Feel free to reach out for collaboration or feedback!
