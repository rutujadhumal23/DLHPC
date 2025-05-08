import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
data = pd.read_csv("letter-recognition.csv")

# Preprocess the data
X = data.drop(columns=['T'])  # Features
y = data['T']  # Target

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Make predictions
predictions = model.predict(X_test)

# Decode the predicted labels
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# Decode the actual labels
actual_labels = label_encoder.inverse_transform(y_test)

# Print some predicted and actual results
print("Some Predicted and Actual Results:")
for i in range(10):
    print("Predicted:", predicted_labels[i], "Actual:", actual_labels[i])


"""
This code is a machine learning project using deep learning to recognize letters of the alphabet based on features. It uses a dataset called letter-recognition.csv, where each row represents a letter (A–Z) and includes numerical features that describe how the letter looks. First, the code loads the dataset using Pandas and separates it into features (X) and labels (y). The labels, which are letters like A, B, C, etc., are converted into numbers using Label Encoding, so the neural network can process them.

Next, the dataset is split into training and test sets. A neural network model is built using Keras with three layers: the first two are hidden layers using ReLU activation, and the final layer uses Softmax to classify the input into one of the 26 letters. The model is compiled using the Adam optimizer and trained using sparse categorical crossentropy loss, which is suitable for multi-class classification with integer labels.

The model is trained for 20 epochs, and during training, 20% of the training data is used for validation. After training, the model is tested on the test set to evaluate its accuracy. Then, predictions are made on the test data, and the numeric predicted labels are converted back to letters using the label encoder. Finally, the code prints out a few examples showing the predicted vs actual letters, helping us understand how well the model performs.

This is a practical example of how deep learning can be used for letter recognition tasks, which is similar to applications like OCR (Optical Character Recognition).
"""


"""
Here's a line-by-line explanation of your code in very easy language:

python
Copy
Edit
import numpy as np
Imports NumPy library, used for working with arrays and numbers.

python
Copy
Edit
import pandas as pd
Imports Pandas, which is used for handling and analyzing data in table form (like Excel).

python
Copy
Edit
from sklearn.model_selection import train_test_split
Imports a function to split the dataset into training and test parts.

python
Copy
Edit
from sklearn.preprocessing import LabelEncoder
Imports LabelEncoder, which converts letter labels (A, B, C, etc.) into numbers (0, 1, 2, ...).

python
Copy
Edit
from tensorflow.keras.models import Sequential
Imports Sequential model from Keras, which means the model will have layers added one after another.

python
Copy
Edit
from tensorflow.keras.layers import Dense
Imports the Dense layer (fully connected layer), used in neural networks.

python
Copy
Edit
from tensorflow.keras.optimizers import Adam
Imports the Adam optimizer, which helps the model learn faster and better.

python
Copy
Edit
data = pd.read_csv("letter-recognition.csv")
Loads the CSV file (dataset) using Pandas. The data contains rows with letter labels and features describing those letters.

python
Copy
Edit
X = data.drop(columns=['T'])  # Features
Drops the 'T' column (which contains the target letter) and keeps only the features (numerical values).

python
Copy
Edit
y = data['T']  # Target
Saves the 'T' column separately, which contains the actual letter labels (A to Z).

python
Copy
Edit
label_encoder = LabelEncoder()
Creates an object to convert letter labels into numbers.

python
Copy
Edit
y_encoded = label_encoder.fit_transform(y)
Transforms the letters (like 'A', 'B') into numeric values (like 0, 1).

python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
Splits the data into training (80%) and testing (20%) sets randomly.

python
Copy
Edit
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])
Builds the neural network model with 3 layers:

First layer: 128 neurons, uses ReLU activation.

Second layer: 64 neurons, also uses ReLU.

Output layer: One neuron for each letter (26 in total), uses Softmax to give probability of each class.

python
Copy
Edit
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
Compiles the model:

Uses Adam optimizer for training.

sparse_categorical_crossentropy is the loss function used for classification with integer labels.

Tracks accuracy during training.

python
Copy
Edit
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
Trains the model:

Runs for 20 times (epochs).

Uses batches of 32 samples.

20% of training data is used for validation during training.

python
Copy
Edit
test_loss, test_accuracy = model.evaluate(X_test, y_test)
Tests the model on the test set and gives the final loss and accuracy.

python
Copy
Edit
print("Test Accuracy:", test_accuracy)
Prints the accuracy on test data.

python
Copy
Edit
predictions = model.predict(X_test)
Makes predictions on the test data. Output is in probabilities for each letter.

python
Copy
Edit
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
Takes the highest probability for each prediction and converts it back from number to letter using the label encoder.

python
Copy
Edit
actual_labels = label_encoder.inverse_transform(y_test)
Converts the actual test labels (numbers) back to letters.

python
Copy
Edit
print("Some Predicted and Actual Results:")
Prints a heading for predicted vs actual results.

python
Copy
Edit
for i in range(10):
    print("Predicted:", predicted_labels[i], "Actual:", actual_labels[i])
Prints the first 10 results showing what the model predicted vs the actual letter in the test set.
"""