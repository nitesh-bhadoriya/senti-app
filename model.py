import numpy as np
import tensorflow as tf
from keras import layers, models, datasets, mixed_precision
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.utils import pad_sequences



'''

# Set random seeds
np.random.seed(7)
tf.random.set_seed(7)

# Mixed precision for faster training
mixed_precision.set_global_policy('mixed_float16')

# Load IMDb dataset
top_words = 5000
(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=top_words)

# Pad sequences
max_review_length = 300
X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)

# Create an efficient data pipeline
batch_size = 128
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# Build a simplified model
embedding_vector_length = 32
model = models.Sequential([
    layers.Embedding(top_words, embedding_vector_length, input_length=max_review_length),
    layers.SpatialDropout1D(0.2),
    layers.Conv1D(128, 5, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset, callbacks=[early_stopping, reduce_lr])

# Save the model
model.save('faster_sentiment_analysis.h5')

# Evaluate the model
scores = model.evaluate(test_dataset, verbose=0)
print("Test Accuracy: %.2f%%" % (scores[1]*100))





'''


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the dataset
top_words = 5000  # Reduced vocabulary size
max_review_length = 500  # Reduced sequence length

(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=top_words)

# Ensure all word indices are within the valid range [0, 4999]
X_train = [[min(word, top_words - 1) for word in review] for review in X_train]
X_test = [[min(word, top_words - 1) for word in review] for review in X_test]

# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_review_length)
X_test = pad_sequences(X_test, maxlen=max_review_length)

# Create data pipelines
batch_size = 128  # Increased batch size
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Build the model
embedding_dim = 64  # Moderate embedding dimension

model = models.Sequential([
    layers.Embedding(top_words, embedding_dim, input_length=max_review_length),
    layers.SpatialDropout1D(0.2),
    layers.LSTM(64),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])


print(model.summary())



# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save('optimized_lstm_sentiment_model.h5')