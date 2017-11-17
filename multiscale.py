#from __future__ import print_function
#import keras
from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
from keras.models import Model
#from keras.layers import Reshape, Conv2D, AveragePooling2D, Dropout, Activation, MaxPooling2D,Input, Dense
from keras.layers import *
#from keras.utils import plot_model 
#import os

input_shape = Input(shape=(84,84,1))
dr_rate = 0.5
# cnn42 = Sequential()
# cnn42.add(Reshape((42,42,4,),input_shape=(84, 84,)))
# # cnn42.summary()
# cnn42.add(Conv2D(6,(10,10), activation='relu', batch_size=4))
# # cnn42.summary()
# cnn42.add(Dropout(dr_rate))

# cnn42.add(AveragePooling2D(pool_size=(4,4),strides=(1,1)))

# cnn42.add(Conv2D(56,(7,7), activation='relu'))
# cnn42.add(Dropout(dr_rate))

# cnn42.add(MaxPooling2D(pool_size=(4,4)))
# cnn42.add(Conv2D(120,(5,5), activation='relu'))
# cnn42.add(Dropout(dr_rate))
# cnn42.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
# cnn42.add(Flatten())
# cnn42.add(Dense(80))
# #cnn42.add(Dense(2)) //only 2 classes for our case
# #cnn42.add(Activation='sigmoid')
# cnn42.summary()

cnn42 = Reshape((42,42,4,))(input_shape)
cnn42 = Conv2D(6,(10,10), activation='relu', batch_size=4)(cnn42)
#cnn42 = Dropout(dr_rate)(cnn42)
cnn42 = AveragePooling2D(pool_size=(4,4),strides=(1,1))(cnn42)
cnn42 = Conv2D(56,(7,7), activation='relu')(cnn42)
cnn42 = Dropout(dr_rate)(cnn42)
cnn42 = MaxPooling2D(pool_size=(4,4),strides=(1,1))(cnn42)
cnn42 = Conv2D(120,(5,5), activation='relu')(cnn42)
cnn42 = Dropout(dr_rate)(cnn42)
cnn42 = MaxPooling2D(pool_size=(2,2),strides=(1,1))(cnn42)
cnn42 = Flatten()(cnn42)
cnn42 = Dense(80)(cnn42)


# cnn14 = Sequential()
# cnn14.add(Reshape((14,14,36,),input_shape=(84, 84,)))
# cnn14.add(Conv2D(60,(8,8), activation='relu'))
# cnn14.add(Dropout(dr_rate))
# cnn14.add(AveragePooling2D(pool_size=(4,4),strides=(1,1)))
# cnn14.add(Flatten())
# cnn14.add(Dense(45))
# #cnn14.add(Dense(2))
# #cnn14.add(Activation='sigmoid')
# cnn14.summary()

cnn14 = Reshape((14,14,36,))(input_shape)
cnn14 = Conv2D(60,(8,8), activation='relu')(cnn14)
cnn14 = Dropout(dr_rate)(cnn14)
cnn14 = MaxPooling2D(pool_size=(4,4),strides=(1,1))(cnn14)
cnn14 = Flatten()(cnn14)
cnn14 = Dense(45)(cnn14)


# cnn84 = Sequential()
# cnn84.add(Conv2D(6,(10,10),input_shape=(84,84,1), activation='relu'))
# cnn84.add(Dropout(dr_rate))
# cnn84.add(AveragePooling2D(pool_size=(4,4),strides=(1,1)))
# cnn84.add(Conv2D(60,(7,7), activation='relu'))
# cnn84.add(Dropout(dr_rate))
# cnn84.add(MaxPooling2D(pool_size=(3,3)))
# cnn84.add(Flatten())
# cnn84.add(Dense(45))
#cnn84.add(Dense(2))
#cnn84.add(Activation='sigmoid')

cnn84 = Conv2D(6,(10,10), activation='relu')(input_shape)
#cnn84 = Dropout(dr_rate)(cnn84)
cnn84 = MaxPooling2D(pool_size=(4,4),strides=(1,1))(cnn84)
cnn84 = (Flatten())(cnn84)
cnn84 = Dense(45)(cnn84)


#merged_model = Concatenate([cnn42,cnn14,cnn84])
merged_model = concatenate([cnn84,cnn42,cnn14])
#final_model = Sequential()                     
#final_model.add(merged_model)                  
# merged_model= keras.layers.Dense(120)(merged_model)
# merged_model= keras.layers.Activation('relu')(merged_model)
#final_model.add(Activation('relu'))
# merged_model= keras.layers.Dropout(dr_rate)(merged_model)
#final_model.add(Dropout(dr_rate))      
# merged_model= keras.layers.Dense(2)(merged_model)
#final_model.add(Dense(2))
# merged_model= keras.layers.Activation('sigmoid')(merged_model)
#final_model.add(Activation('sigmoid'))

final_model = Dropout(0.25)(merged_model)
final_model = Dense(120)(final_model)
final_model = Dropout(dr_rate)(final_model)
final_model = Dense(2)(final_model)
final_model = Activation('sigmoid')(final_model)

final_model = Model(input=input_shape, outputs=final_model)
#plot_model(final_model,to_file='diagram.png')

final_model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

final_model.summary()
train_datagen = ImageDataGenerator(rescale=1. / 255,horizontal_flip=1, vertical_flip=1)
train_generator = train_datagen.flow_from_directory('train', target_size=(84,84), batch_size=3, 
                                                    class_mode='categorical', color_mode='grayscale')

validation_datagen = ImageDataGenerator(rescale=1. / 255,horizontal_flip=1, vertical_flip=1)
validation_generator = train_datagen.flow_from_directory('validation', target_size=(84,84), batch_size=3,
                                                     class_mode='categorical', color_mode='grayscale')

test_datagen = ImageDataGenerator(rescale=1. / 255,horizontal_flip=1, vertical_flip=1)
test_generator = test_datagen.flow_from_directory('test', target_size=(84,84), batch_size=3,
                                                     class_mode='categorical', color_mode='grayscale')
#final_train_generator = zip(train_generator, train_generator, train_generator)
#final_test_generator  = zip(test_generator, test_generator, test_generator)
final_model.fit_generator(train_generator, steps_per_epoch=60, epochs=20,
                             validation_data=validation_generator, verbose=1, workers=1, validation_steps=30)
final_model.save_weights('multi.h5')
#score = model.evaluate_generator(final_test_generator)
score = final_model.evaluate_generator(test_generator, 1, workers=1)
print()
print(score)
print()
#print("Correct:", correct, " Total: ", len(validation_generator.filenames))
print("Loss: ", score[0], "Accuracy: ", score[1])