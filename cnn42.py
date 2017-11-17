from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import *
from keras.layers import Conv2D, MaxPooling2D
import os

cnn42 = Sequential()
cnn42.add(Conv2D(6,(10,10), input_shape=(42, 42,1), activation='relu', data_format="channels_last"))
cnn42.add(Dropout(0.5))

cnn42.add(AveragePooling2D(pool_size=(4,4),strides=(1,1)))

cnn42.add(Conv2D(56,(7,7), activation='relu'))
cnn42.add(Dropout(0.5))

cnn42.add(MaxPooling2D(pool_size=(4,4), data_format='channels_last'))
cnn42.add(Conv2D(120,(5,5), activation='relu'))
cnn42.add(Dropout(0.5))
cnn42.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
cnn42.add(Flatten())
cnn42.add(Dense(80))
cnn42.add(Dense(2)) #only 2 classes for our case
cnn42.add(Activation('sigmoid'))

cnn42.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory('train', target_size=(42,42), batch_size=1, 
                                                    class_mode='categorical', color_mode='grayscale')

validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = train_datagen.flow_from_directory('validation', target_size=(42,42), batch_size=1,
                                                     class_mode='categorical', color_mode='grayscale')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory('test', target_size=(42,42), batch_size=1,
                                                     class_mode='categorical', color_mode='grayscale')

#final_train_generator = zip(train_generator, train_generator, train_generator)
#final_test_generator  = zip(test_generator, test_generator, test_generator)
cnn42.fit_generator(train_generator, steps_per_epoch=24, epochs=100,
                             validation_data=validation_generator, verbose=1, workers=3, validation_steps=24)
cnn42.save_weights('cnn42_rescale.h5')
#score = model.evaluate_generator(final_test_generator)
score = cnn42.evaluate_generator(test_generator, 1, workers=1)
print()
print(score)
print()
#print("Correct:", correct, " Total: ", len(validation_generator.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])