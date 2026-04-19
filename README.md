# 🚨 SirenSense: TinyML Emergency Siren Detection

SirenSense is a real-time TinyML system that detects emergency vehicle sirens (e.g. ambulance, fire truck) using on-device audio classification.

The model runs directly on an Arduino Nano 33 BLE Sense, enabling low-latency, energy-efficient inference without relying on cloud processing.

---

## 🚀 Motivation

In real-world environments, detecting emergency sirens quickly is critical for improving response times and enabling smarter traffic systems.

Traditional cloud-based solutions introduce latency and depend on connectivity. SirenSense addresses this by performing on-device inference using TinyML, making it faster, more efficient, and privacy-preserving.

---

## 🛠️ System Overview

1. Microphone captures real-time audio  
2. Audio is preprocessed into MFCC features  
3. A lightweight CNN model performs classification  
4. Output is indicated using onboard LEDs:  
   - 🔴 Red → Ambulance  
   - 🔵 Blue → Fire Truck  
   - ⚪ Off → Ambient noise  

---

## ⚙️ Tech Stack

- Python (model training)
- TensorFlow / TensorFlow Lite
- TinyML
- Arduino
- Audio signal processing (MFCC)

---

## 📊 Model Details

- Input: MFCC feature map (13 × 400)  
- Model: Lightweight CNN with separable convolution  
- Parameters: ~2000  
- Accuracy: ~95% (after INT8 quantization)  
- Optimization: Post-training quantization (INT8)

The quantized model significantly reduces size and latency while maintaining strong performance, making it suitable for embedded deployment.

---

## 🧪 Key Learnings

- Designing ML systems under hardware constraints  
- Trade-offs between accuracy and efficiency  
- End-to-end pipeline: data → model → deployment  
- Real-time inference on embedded devices  

---

## ⚠️ Limitations

- Limited to 3 classes (ambulance, fire truck, ambient noise)  
- Some confusion between similar siren types  
- Dataset size relatively small  

---

## 🔮 Future Improvements

- Expand dataset for better generalization  
- Add more sound classes (e.g. police sirens)  
- Improve robustness in noisy environments  
- Optimize power consumption for continuous use  

---

## 👩‍💻 Author

Sudhishna Janavi
