# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d

# 1. Ensure you are using Theano backend
import os
os.environ['KERAS_BACKEND'] = 'theano'

# 2. Import libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

# 3. Set random seed (for reproducibility)
np.random.seed(123)
 
# 4. Load pre-shuffled MNIST data into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
 
# 5. Preprocess input data
X_train_reshape = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test_reshape = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train_reshape = X_train.astype('float32')
X_test_reshape = X_test.astype('float32')
X_train_reshape /= 255
X_test_reshape /= 255
 
# 6. Preprocess class labels
Y_train_categorical = np_utils.to_categorical(Y_train, 10)
Y_test_categorical = np_utils.to_categorical(Y_test, 10)
 
# 7. Define model architecture
model = Sequential()
 
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
model.fit(X_train_reshape[:100], Y_train_categorical[:100], 
          batch_size=32, epochs=1, verbose=1)
 
# 10. Evaluate model on test data
loss, accuracy = model.evaluate(X_test_reshape[:100], Y_test_categorical[:100], verbose=0)

# 11. Check the label that has been predicted
predicted_labels = model.predict_classes(X_test_reshape[:100])

# 12. Check the label that has been predicted incorrectly
incorrect_labels = np.nonzero(model.predict_classes(X_test_reshape[:100]).reshape((-1,)) != Y_test[:100])

############################################################################

# 13. Show a sample from the mnist dataset
image_to_show = 72
from_group = 'test' # put 'train' or 'test'

if from_group == 'train':
    plt.imshow(X_train[image_to_show])
    print('Ground truth label: ',Y_train[image_to_show])
else:
    plt.imshow(X_test[image_to_show])
    print('Ground truth label: ',Y_test[image_to_show])
    if len(predicted_labels)>image_to_show:
        print('Predicted label: ',predicted_labels[image_to_show])