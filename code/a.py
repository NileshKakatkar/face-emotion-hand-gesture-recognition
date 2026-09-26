from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os
import numpy as np
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
label_map = {label:num for num, label in enumerate(actions)}
# print(label_map)
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        try:
            res = np.load(os.path.join(DATA_PATH, action, "{}.npy".format(sequence)),allow_pickle=True)
            sequences.append(res)
            labels.append(label_map[action])
        except:
            print('')

X = np.array(sequences)
y = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# ... (your previous code)

# Ensure log directory exists for TensorBoard
log_dir = os.path.join('Logs')
os.makedirs(log_dir, exist_ok=True)
tb_callback = TensorBoard(log_dir=log_dir)

# Create a Sequential model

model = Sequential([
    Flatten(input_shape=(21, 2), name='flatten_2'),  # Flattens input of shape (21, 2) to (None, 42)
    Dense(64, activation='relu', name='dense_11'),  # Dense layer with 64 units
    Dense(128, activation='relu', name='dense_12'),  # Dense layer with 128 units
    Dense(512, activation='relu', name='dense_13'),  # Dense layer with 512 units
    Dense(64, activation='relu', name='dense_14'),  # Dense layer with 64 units
    Dense(32, activation='relu', name='dense_15'),  # Dense layer with 32 units
    Dense(16, activation='softmax', name='dense_16')  # Dense layer with 10 units and softmax activation
])

# Compile the model and specify loss and optimizer
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
model.summary() 

# Train the model
model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback], validation_data=(X_test, y_test))

# Save the model
model.save('../models/hand_gesture')