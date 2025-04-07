
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
import os

# Ensure nltk data is available
nltk.download('punkt')

# Load and prepare the text
with open("sherlock.txt", "r", encoding="utf-8") as f:
    text = f.read()

sentences = nltk.sent_tokenize(text)
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_seq_len = max([len(x) for x in input_sequences])

# Load trained model
model = tf.keras.models.load_model("my_model.keras")

# Text generation function
def generate_text(seed_text, next_words, temperature):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
        predictions = model.predict(token_list, verbose=0)[0]

        predictions = np.asarray(predictions).astype('float64')
        predictions = np.log(predictions + 1e-10) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, predictions, 1)
        output_word = tokenizer.index_word[np.argmax(probas)]

        seed_text += " " + output_word
    return seed_text

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Sherlock Holmes Text Generator (LSTM)")
    seed = gr.Textbox(label="Enter Seed Text", value="Sherlock Holmes was")
    words = gr.Slider(minimum=10, maximum=100, value=30, step=5, label="Number of Words to Generate")
    temperature = gr.Slider(minimum=0.2, maximum=1.5, value=0.8, step=0.1, label="Creativity (Temperature)")
    output = gr.Textbox(label="Generated Text")

    generate_button = gr.Button("Generate Text")

    generate_button.click(fn=generate_text, inputs=[seed, words, temperature], outputs=output)

demo.launch()
