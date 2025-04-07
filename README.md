# 🧠 LSTM Text Generator

This project is a simple text generation model using LSTM, trained on *The Adventures of Sherlock Holmes*.  
It takes a text prompt and predicts the next words based on the training dataset.

## 🔧 Tech Stack
- TensorFlow / Keras
- NLTK
- Gradio (for UI)
- Hugging Face Spaces (for deployment)

## 🚀 Try It Live
👉 [Check the live demo on Hugging Face](https://huggingface.co/spaces/bhavya0825/text_generation_using_lstm)

## 📁 Files
- `app.py`: Gradio interface to run the model
- `model.h5`: Trained LSTM model
- `sherlock.txt`: Text data used for training
- `requirements.txt`: Python dependencies

## 🛠 Run Locally

```bash
pip install -r requirements.txt
python app.py
