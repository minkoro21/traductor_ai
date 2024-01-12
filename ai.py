import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Charger le jeu de données depuis le fichier CSV
data = pd.read_csv('/content/train.csv')

# Reste du code reste inchangé...

# Reste du code reste inchangé...

# Supprimer les lignes avec des valeurs manquantes
data = data.dropna()

# Diviser les données en ensembles d'entraînement et de validation
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Tokenization du texte
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data['Context'] + train_data['Response'])

# Taille du vocabulaire
vocab_size = len(tokenizer.word_index) + 1

# Encoder les séquences de texte en séquences d'entiers
train_context_sequences = tokenizer.texts_to_sequences(train_data['Context'])
train_response_sequences = tokenizer.texts_to_sequences(train_data['Response'])

val_context_sequences = tokenizer.texts_to_sequences(val_data['Context'])
val_response_sequences = tokenizer.texts_to_sequences(val_data['Response'])

# Paddings des séquences pour qu'elles aient la même longueur
max_sequence_length = max(max(len(seq) for seq in train_context_sequences),
                         max(len(seq) for seq in train_response_sequences))

train_context_sequences = pad_sequences(train_context_sequences, maxlen=max_sequence_length, padding='post')
train_response_sequences = pad_sequences(train_response_sequences, maxlen=max_sequence_length, padding='post')

val_context_sequences = pad_sequences(val_context_sequences, maxlen=max_sequence_length, padding='post')
val_response_sequences = pad_sequences(val_response_sequences, maxlen=max_sequence_length, padding='post')

# Création du jeu de données TensorFlow
train_dataset = tf.data.Dataset.from_tensor_slices((train_context_sequences, train_response_sequences))
val_dataset = tf.data.Dataset.from_tensor_slices((val_context_sequences, val_response_sequences))

# Paramètres pour l'entraînement du modèle
batch_size = 64
train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)

# Maintenant, intégrez ces données dans votre modèle
num_layers = 4
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = vocab_size  # Taille du vocabulaire d'entrée
target_vocab_size = vocab_size  # Taille du vocabulaire de sortie

dropout_rate = 0.1

# Couche d'encodage
inputs = tf.keras.Input(shape=(None,))
embedding_layer = tf.keras.layers.Embedding(input_vocab_size, d_model)
enc_output = embedding_layer(inputs)
enc_output = tf.keras.layers.Dropout(dropout_rate)(enc_output)

for _ in range(num_layers):
    enc_output = tf.keras.layers.MultiHeadAttention(num_heads, d_model)(enc_output, enc_output, enc_output)
    enc_output = tf.keras.layers.Dropout(dropout_rate)(enc_output)
    enc_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(enc_output)
    enc_output = tf.keras.layers.Dense(dff, activation="relu")(enc_output)
    enc_output = tf.keras.layers.Dropout(dropout_rate)(enc_output)
    enc_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(enc_output)

# Couche de décodage
outputs = tf.keras.layers.Dense(target_vocab_size, activation="softmax")(enc_output)

# Créez le modèle complet
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compilez et entraînez le modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=10, validation_data=val_dataset)
