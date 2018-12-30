import tensorflow as tf
import pandas as pd
import numpy as np
import data as dt

class Module():


    def __init__(self):
        pass

    def CNN_model(self):
        # Input:96x96x3 --> Conv:26x26x16 --> Conv:24x24x32 --> Pooling:12x12x32 --> Hidden:256 --> Hidden:64 --> Hidden:10 --> Softmax --> Output
        # Loss Func: CrossEntropy  /  Optimizer: Adam Optimizer
        # Etc: BatchNormalization
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), padding='valid', input_shape=(96, 96, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
        ])
        model.compile(optimizer='adam', loss=tf.keras.backend.binary_crossentropy, metrics=['accuracy'])
        return model

    def train(self, model, train, label, epoch, batch_size):
        hist = model.fit(train, label, epochs=epoch, batch_size=batch_size)
        return hist

    def print_train_result(self, train):
        print('## train loss and acc ##')
        print(train.history['loss'])
        print(train.history['acc'])

    def model_predict(self, model, X_test):
        return model.predict(X_test)

    def evaluate_prediction(self, prediction, name):
        result = []
        for i, predict in enumerate(prediction):
            if predict >= 0.5:
                result.append([name[i], 1])
            else:
                result.append([name[i], 0])
        return result

    def save_csv(self, result, filename):
        df = pd.DataFrame(data=np.array(result), columns=['id', 'label'])
        df.to_csv(filename+".csv", header=True, index=False)

if __name__ == "__main__":
    dt1 = dt.DataLoad()
    dt1.open_csv('train_labels.csv')
    dt1.search_file('E://Data/Histopathologic Cancer Detection Data/train')
    X_label = []
    X_train = np.array(dt1.image_data_list)
    X_name = np.array(dt1.image_name_list)
    for i in range(len(X_name)):
        X_label.append(int(dt1.label_dic[X_name[i]]))

    dt2 = dt.DataLoad()
    dt2.search_file('E://Data/Histopathologic Cancer Detection Data/test')
    X_test = np.array(dt2.image_data_list)
    X_test_name = np.array(dt2.image_name_list)

    #   Convolutional Neural Network Train and Evaluate
    module = Module()
    model = module.CNN_model()
    train = module.train(model=model, train = X_train, label=X_label, epoch=30, batch_size=1000)
    module.print_train_result(train)
    predictions = module.model_predict(model, X_test)
    result = module.evaluate_prediction(predictions, X_test_name)
    module.save_csv(result, 'result')
