import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

# Define the number of words to consider as features
max_features = 10000

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)

# Set the maximum length for review sequences
maxlen = 500

# Preprocess the data
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)

# Define the model
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
model.fit(train_data, train_labels, epochs=1, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print("Test Accuracy:", test_accuracy)



"""
This Python code is for building a sentiment analysis model using the IMDB movie reviews dataset. The goal is to classify reviews as positive or negative using a deep learning model with Keras.

First, the IMDB dataset is loaded, where each review is already converted into a sequence of numbers (each number represents a word). The code keeps only the top 10,000 most frequent words to reduce complexity. Since reviews are of different lengths, all sequences are made the same size (500 words) using padding, so the model can process them properly.

The model used here is simple and has three layers:

Embedding layer – It converts word indices into dense vectors of fixed size (128). This helps the model understand the meaning of words better.

Flatten layer – It reshapes the data into a format suitable for the next layer.

Dense layer – This is the output layer with a sigmoid activation function to give a result between 0 and 1 (0 = negative review, 1 = positive review).

The model is compiled using the Adam optimizer and binary cross-entropy loss since it’s a binary classification problem. It is then trained on the data for 1 epoch, using 80% of the training data for training and 20% for validation. Finally, the model is tested on the test data, and the test accuracy is printed.

This code is a basic example of how deep learning can be used to analyze text and predict sentiment from movie reviews.
"""

"""
Here’s the line-by-line explanation of your code in very easy language:

python
Copy
Edit
import numpy as np
This line imports NumPy, a library used for working with arrays and numerical operations.

python
Copy
Edit
from tensorflow.keras.datasets import imdb
Imports the IMDB dataset from Keras. This dataset contains 50,000 movie reviews labeled as positive or negative.

python
Copy
Edit
from tensorflow.keras.preprocessing.sequence import pad_sequences
Imports a function to make all input sequences (reviews) of the same length by adding padding.

python
Copy
Edit
from tensorflow.keras.models import Sequential
Imports the Sequential model from Keras, which allows us to build a model layer-by-layer.

python
Copy
Edit
from tensorflow.keras.layers import Dense, Embedding, Flatten
Imports the layers used in the model:

Dense: Fully connected layer.

Embedding: Converts word indices into vectors.

Flatten: Converts multi-dimensional data into 1D for the Dense layer.

python
Copy
Edit
max_features = 10000
We only keep the top 10,000 most frequent words from the dataset.

python
Copy
Edit
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=max_features)
Loads the training and test data.

Each review is a sequence of numbers (word indexes), and each label is 0 (negative) or 1 (positive).

Only words in the top 10,000 are kept.

python
Copy
Edit
maxlen = 500
Sets the maximum length of each review to 500 words.

python
Copy
Edit
train_data = pad_sequences(train_data, maxlen=maxlen)
test_data = pad_sequences(test_data, maxlen=maxlen)
Pads or trims all reviews to make them exactly 500 words long. This helps in giving input of uniform size to the model.

python
Copy
Edit
model = Sequential()
Creates a Sequential model, meaning the layers will be added one after another.

python
Copy
Edit
model.add(Embedding(max_features, 128, input_length=maxlen))
Adds an Embedding layer:

Input size is 10,000 (vocabulary).

Each word will be converted into a 128-dimensional vector.

Input length is 500 (each review has 500 words).

python
Copy
Edit
model.add(Flatten())
Flattens the output from the Embedding layer into a 1D array to feed into the Dense layer.

python
Copy
Edit
model.add(Dense(1, activation='sigmoid'))
Adds a Dense (fully connected) output layer:

Only 1 output neuron.

sigmoid activation is used to output a value between 0 and 1 (probability of positive review).

python
Copy
Edit
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
Compiles the model:

adam: Optimizer that adjusts weights efficiently.

binary_crossentropy: Used for binary classification (positive/negative).

accuracy: Used to evaluate the model’s performance.

python
Copy
Edit
batch_size = 32
Sets the number of samples processed before the model is updated once.

python
Copy
Edit
model.fit(train_data, train_labels, epochs=1, batch_size=batch_size, validation_split=0.2)
Trains the model:

Runs for 1 epoch (1 complete pass through the training data).

Uses batch size of 32.

20% of training data is used for validation to check how well the model is learning.

python
Copy
Edit
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
Evaluates the model on the test data.

Returns loss and accuracy on unseen data.

python
Copy
Edit
print("Test Accuracy:", test_accuracy)
Prints the accuracy of the model on test data.
"""