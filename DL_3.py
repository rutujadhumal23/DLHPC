import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('fashion-MNIST.csv')

# Split features and labels
X = data.iloc[:, 1:].values.astype('float32')
y = data.iloc[:, 0].values.astype('int32')

# Preprocess the data
X /= 255.0
X = X.reshape(-1, 28, 28, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encoding
num_classes = len(np.unique(y))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Make predictions
predictions = model.predict(X_test)

# Print actual vs predicted classes
print("Some actual and predicted classes:")
for i in range(10):
    actual_class = np.argmax(y_test[i])
    predicted_class = np.argmax(predictions[i])
    print(f"Sample {i+1}: Actual cloth class: {actual_class}, Predicted cloth class: {predicted_class}")


"""
This code is a Convolutional Neural Network (CNN) model built using TensorFlow and Keras to classify clothing images from the Fashion MNIST dataset. The dataset is first loaded using pandas, where each image is a 28x28 grayscale picture and each label represents a type of clothing (like T-shirt, trousers, etc.). The features (X) are normalized to values between 0 and 1 and reshaped to fit the CNN input format. The labels (y) are one-hot encoded for multiclass classification. The model is then built using several convolutional layers (Conv2D), each followed by max-pooling (MaxPooling2D) to reduce dimensions and extract features. A dropout layer is added to avoid overfitting. Finally, a fully connected (Dense) layer with softmax activation outputs the probability for each class. The model is trained for 10 epochs, validated using a split of the training data, and then evaluated on the test set. The accuracy is printed, and predictions for the first 10 samples are compared with actual labels to check how well the model performs.
"""

"""
Sure! Here's a simple line-by-line explanation of your code in very easy language:

python
Copy
Edit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
👉 We are importing useful libraries:

numpy for numerical calculations

pandas to handle CSV data

matplotlib.pyplot (though not used here) is for plotting graphs (optional)

python
Copy
Edit
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
👉 These are imports from machine learning and deep learning:

train_test_split helps split the data for training and testing

Sequential helps build our model layer by layer

Layers like Conv2D, MaxPooling2D, etc., are used to create a CNN

to_categorical helps convert labels into a format the model understands

python
Copy
Edit
data = pd.read_csv('fashion-MNIST.csv')
👉 Loads the Fashion MNIST dataset from a CSV file.

python
Copy
Edit
X = data.iloc[:, 1:].values.astype('float32')
y = data.iloc[:, 0].values.astype('int32')
👉 Splits the data:

X contains image pixel values

y contains the label (type of clothing)
We convert them to the correct number format.

python
Copy
Edit
X /= 255.0
X = X.reshape(-1, 28, 28, 1)
👉 Normalize the pixel values (from 0–255 to 0–1)
👉 Reshape the images to 28x28 with 1 color channel (grayscale)

python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
👉 Splits the dataset into:

80% training data

20% test data

python
Copy
Edit
num_classes = len(np.unique(y))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
👉 Counts how many unique clothing types (classes)
👉 Converts labels into one-hot encoding format (e.g., 3 becomes [0,0,0,1,0,...])

python
Copy
Edit
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
👉 Builds a CNN model:

3 Convolution + Pooling layers to extract features from images

Flatten() turns the 2D features into 1D

Dense is a fully connected layer

Dropout randomly ignores some neurons to avoid overfitting

Last Dense layer gives the final prediction using softmax

python
Copy
Edit
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
👉 Prepares the model:

adam is the optimizer

categorical_crossentropy is the loss function used for multi-class classification

accuracy will be tracked while training

python
Copy
Edit
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
👉 Trains the model:

for 10 cycles (epochs)

using 128 images at a time (batch size)

20% of training data is used to validate the model during training

python
Copy
Edit
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')
👉 Tests the model on unseen data and prints how accurate it is.

python
Copy
Edit
predictions = model.predict(X_test)
👉 Model predicts labels for the test images.

python
Copy
Edit
print("Some actual and predicted classes:")
for i in range(10):
    actual_class = np.argmax(y_test[i])
    predicted_class = np.argmax(predictions[i])
    print(f"Sample {i+1}: Actual cloth class: {actual_class}, Predicted cloth class: {predicted_class}")
👉 Compares the actual and predicted clothing types for the first 10 test images.
"""