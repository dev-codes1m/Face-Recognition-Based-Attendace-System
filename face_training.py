from keras.layers import Input,Lambda,Dense,Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import  glob
import matplotlib.pyplot as plt

IMAGE_SIZE=[224,224]

train_path = r'D:\Deep Learning Project\Face Recognition Model\images\train/'
valid_path = r'D:\Deep Learning Project\Face Recognition Model\images\Validation/'

# Add Preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3],weights='imagenet',include_top=False)

# Dont train existing weights

for layers in vgg.layers:
    layers.trainable = False

# useful for getting number
folders = glob(r'D:\Deep Learning Project\Face Recognition Model\images\train/*')

# Making Our own layers - you can add more if u want

x = Flatten()(vgg.output)
prediction = Dense(len(folders),activation='softmax')(x)

model = Model(vgg.input,prediction)
model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1./255,
                shear_range=0.2,
                zoom_range= 0.2,
                horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(r'D:\Deep Learning Project\Face Recognition Model\images\train/',target_size=(224,224),batch_size=8,class_mode = 'categorical')
test_set = test_datagen.flow_from_directory(r'D:\Deep Learning Project\Face Recognition Model\images\Validation/',target_size=(224,224),batch_size=8,class_mode = 'categorical')

r = model.fit_generator(
    training_set,validation_data=test_set,epochs=5,
    steps_per_epoch=len(training_set),validation_steps=len(test_set)
)

plt.plot(r.history['loss'],label= 'train loss',color='blue')
plt.plot(r.history['val_loss'],label = 'val loss',color='orange')
plt.legend()
# plt.show()
plt.savefig(r'D:\Deep Learning Project\Face Recognition Model\Loss_Val')

plt.plot(r.history['accuracy'],label= 'train acc',color='yellow')
plt.plot(r.history['val_accuracy'],label = 'val acc',color='green')
plt.legend()
# plt.show()
plt.savefig(r'D:\Deep Learning Project\Face Recognition Model\ACC_Val')

model.save(r'D:\Deep Learning Project\Face Recognition Model\face_recognition.h5')



