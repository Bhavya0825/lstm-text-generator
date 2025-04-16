import gradio as gr
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("text_generation_lstm_model.keras")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

max_seq_len = model.input_shape[1]
total_words = model.output_shape[-1]

# Text generation function
def generate_text(seed_text, next_words=30, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len, padding='pre')

        predictions = model.predict(token_list, verbose=0)[0]
        predictions = np.log(predictions + 1e-8) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        next_index = np.random.choice(range(total_words), p=predictions)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                output_word = word
                break

        if output_word == "":
            break
        seed_text += " " + output_word
    return seed_text

# Gradio interface
def interface(seed_text, num_words, temperature):
    return generate_text(seed_text, next_words=num_words, temperature=temperature)

demo = gr.Interface(
    fn=interface,
    inputs=[
        gr.Textbox(lines=2, label="Seed Text", placeholder="Start your Sherlock Holmes story..."),
        gr.Slider(10, 100, value=30, step=1, label="Number of Words"),
        gr.Slider(0.2, 1.5, value=0.8, step=0.1, label="Creativity (Temperature)")
    ],
    outputs=gr.Textbox(lines=10, label="Generated Text"),
    title="🔮 Sherlock Holmes Text Generator",
    description="Generate Sherlock-style text using a trained LSTM model.",
)

demo.launch(share=True)

