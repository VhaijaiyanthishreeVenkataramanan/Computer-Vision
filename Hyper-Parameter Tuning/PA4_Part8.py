from __future__ import print_function
import keras
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

batch_size = 128
num_classes = 10
epochs = 30


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=5, strides=1,
                 activation='relu',
                 input_shape=x_train[0].shape))
model.add(AveragePooling2D(pool_size=2, strides=2))
model.add(Conv2D(filters=16, kernel_size=5, strides=1, activation='relu', input_shape=(14, 14, 6)))
model.add(AveragePooling2D(pool_size=2, strides=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units = 120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 84, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

sgd = keras.optimizers.SGD(lr=0.001)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])
hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print(score)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


train_acc = hist.history['acc']
test_acc = hist.history['val_acc']
loss_or_cost = hist.history['loss']
xc = range(epochs)
plt.figure(1, figsize=(7, 5))
plt.plot(xc, train_acc)
plt.plot(xc, test_acc)
plt.xlabel('# of Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracies vs epoch on Fashion MNIST Dataset using LeNet-5 Model and SGD optimizer')
plt.grid(True)
plt.legend(['Train', 'Test'], loc=4)
plt.style.use(['classic'])

plt.figure(2, figsize=(7, 5))
plt.plot(xc, loss_or_cost)
plt.xlabel('# of Epochs')
plt.ylabel('Loss or Cost')
plt.title('Loss(or)Cost vs epoch on Fashion MNIST Dataset using LeNet-5 Model and SGD optimizer')
plt.grid(True)
plt.style.use(['classic'])