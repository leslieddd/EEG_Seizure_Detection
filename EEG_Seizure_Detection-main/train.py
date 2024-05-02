from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define 2L-LSTM model
def lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=138, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(units=64))
    model.add(Dropout(0.5))  # Using the specified dropout probability
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    return model

# Define input shape
num_time_steps = 10
num_features = 15*18 

input_shape = (num_time_steps, num_features)  # Ten features extracted from five frequency bands of 23-channel EEG signal

# Create 2L-LSTM model
model_lstm = lstm_model(input_shape)
model_lstm.summary()

# Compile the model
model_lstm.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (adjust epochs according to the provided specifications)
patient_num = 1
from process import process
x,y = process(patient_num, 30, 120, num_time_steps)
#print(len(x))
#print(len(y))

# Define the percentage of data to be used for testing (e.g., 20% --> 0.2)
test_percent = 0.2

# Calculate the number of samples to be used for testing
num_test_samples = int(test_percent * len(x))

# Split the data into training and testing sets
X_train = x[:-num_test_samples]
X_test = x[-num_test_samples:]

y_train = y[:-num_test_samples]
y_test = y[-num_test_samples:]

model_lstm.fit(X_train, y_train, batch_size=32, epochs=30)

y_test_pred = model_lstm.predict(X_test)
y_test_pred = np.argmax(y_test_pred, axis=1)
y_test_true = np.argmax(y_test, axis=1)

#cm = confusion_matrix(y_test_true, y_test_pred)
#print(cm)


# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()
