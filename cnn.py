# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'C:\Users\Marcelo\Documents\Python Scripts\dataset\training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'C:\Users\Marcelo\Documents\Python Scripts\dataset\test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, # total number of images at the training set
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000) # number of images at test set

# save the deep learning model’s weights

from keras.models import load_model

classifier.save('dog_cat_model.h5')  # creates a HDF5 file 'my_model.h5'

del classifier  # deletes the existing model

# returns a compiled model
# identical to the previous one
# model = load_model('dog_cat_model.h5')
classifier = load_model('dog_cat_model.h5')

# Part 3 - Making new predictions
# Zizou_imgTest1.jpeg
# Zizou_imgTest2.jpeg
# IMG_0004.jpg
# cat_or_dog_2.jpg
# cat_or_dog_1.jpg
# Test_Img_1.jpeg
# Test_Img_2.jpeg
# Test_Img_3.jpeg
# Test_Img_4.jpeg
# Test_Img_5.jpeg
# Test_Img_6.jpeg

import numpy as np
from keras.preprocessing import image
test_image = image.load_img(r'C:\Users\Marcelo\Documents\Python Scripts\dataset\single_prediction\Test_Img_6.jpeg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'