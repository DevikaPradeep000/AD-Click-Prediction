# 🎯 AdPredict.AI: GPU-Accelerated Click Prediction

AdPredict.AI is a state-of-the-art machine learning web application designed to predict the probability of a user clicking on an advertisement. By leveraging the **NVIDIA RAPIDS** ecosystem, the platform provides lightning-fast training on massive datasets (500,000+ rows) while offering a premium, user-friendly interface for real-time analytics.

---

## ✨ Key Features

- **🚀 Hybrid GPU/CPU Acceleration**: Automatically detects and uses **RAPIDS (cuDF & cuML)** for GPU-accelerated training. Seamlessly falls back to Scikit-Learn on CPU-only systems.
- **🧠 Explainable AI (XAI)**: Provides detailed intelligence reports for every prediction, explaining the "Why" behind the score.
- **📊 Interactive Analytics Dashboard**: Visualize model performance and feature importance (e.g., how much Age vs. Device Type affects clicks) using Chart.js.
- **💾 Model Persistence**: Trained models are saved to disk, allowing for instant predictions without needing to retrain on every restart.
- **🎨 Premium UI/UX**: Modern glassmorphism design with a dark aesthetic, responsive layouts, and smooth animations.
- **🔗 URL Intelligence**: Automatically extracts ad category and link length from pasted URLs.

---

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Machine Learning**: NVIDIA RAPIDS (cuML/cuDF), Scikit-Learn
- **Database**: SQLite3
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (ES6+)
- **Visualization**: Chart.js

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate Data
Create the synthetic dataset of 500,000 samples:
```bash
python generate_dataset.py
```

### 3. Run the App
Start the Flask development server:
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.

---

## 🔮 Future Scope & Roadmap

- **🤖 AI Campaign Advisor**: A generative AI agent to suggest ad optimizations.
- **📡 Real-Time Bidding (RTB)**: Integration with live ad exchanges for automated bidding.
- **🔄 A/B Testing Simulator**: Simulate and compare multiple ad variations before going live.
- **🌐 Cross-Platform API**: Plug-and-play integration for Google Ads and Meta Ads Managers.

---

## 📝 License
This project is developed for educational purposes as part of a Final Academic Review.

---
*Developed with 💙 using Python & RAPIDS.*
