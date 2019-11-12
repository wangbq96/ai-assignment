from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
import time
import csv


class History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.loss = []
        self.acc = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        self.loss.append(logs.get("loss"))
        self.acc.append(logs.get("accuracy"))

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# build model
model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))

history = History()
sgd = optimizers.SGD(lr=0.001, momentum=0.9)
model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=15, verbose=1, validation_data=(x_test, y_test), callbacks=[history])
# score = model.evaluate(x_test, y_test)
# print('Test Loss:', score[0])
# print('Test accuracy:', score[1])
# print(history.times)
# print(history.loss)
# print(history.acc)

with open("record.csv", "w", newline="") as f:
    f_csv = csv.writer(f)
    headers = ["time", "loss", "acc"]
    f_csv.writerow(headers)
    for i in range(len(history.times)):
        row = [history.times[i], history.loss[i], history.acc[i]]
        f_csv.writerow(row)