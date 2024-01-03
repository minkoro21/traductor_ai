from tensorflow import keras
import tensorflow as tf
import numpy as np

data = [("Hello", "Bonjour"), ("Goodbye", "Au revoir"), ("Thank you", "Merci")]

texts, labels = zip(*data)

tokenizer_input = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer_input.fit_on_texts(texts)

tokenizer_output = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer_output.fit_on_texts(labels)

X = tokenizer_input.texts_to_sequences(texts)
y = tokenizer_output.texts_to_sequences(labels)

max_len = max(len(seq) for seq in X + y)

X = keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, padding='post')
y = keras.preprocessing.sequence.pad_sequences(y, maxlen=max_len, padding='post')

num_classes = len(tokenizer_output.word_index) + 1
y = keras.utils.to_categorical(y, num_classes=num_classes)

model = keras.Sequential(
    [
        keras.layers.Embedding(input_dim=len(tokenizer_input.word_index) + 1, output_dim=10, input_length=max_len),
        keras.layers.LSTM(50, return_sequences=True),  # Change here to return sequences
        keras.layers.Dense(num_classes, activation='softmax')
    ]
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=50, verbose=2)
