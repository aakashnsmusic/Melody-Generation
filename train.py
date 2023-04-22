from preprocess import generate_training_sequences,SEQUENCE_LENGTH
import keras
from tensorflow.keras.callbacks import TensorBoard
import tensorflow
import time

OUTPUT_UNITS=38
LOSS="sparse_categorical_crossentropy"
LEARNING_RATE=0.001
NUM_UNITS=[256]
EPOCHS=50
BATCH_SIZE=64
SAVE_MODEL_PATH="tb_model1.h5"
NAME="loss-{}".format(int(time.time()))


def build_model(output_units,num_units,loss,learning_rate):
    #create model architecture
    #None allows you to have as many time series inputs as you want
    input=keras.layers.Input(shape=(None,output_units))
    x=keras.layers.LSTM(num_units[0],return_sequences=True)(input)
    #x=keras.layers.LSTM(num_units[0])(input)
    x=keras.layers.Dropout(0.2)(x)
    x=keras.layers.LSTM(128)(x)
    x=keras.layers.Dropout(0.1)(x)
    output=keras.layers.Dense(output_units,activation="softmax")(x)
    model=keras.Model(input,output)
    #compile model
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  metrics=["accuracy"])
    model.summary()
    return model

def train(output_units=OUTPUT_UNITS,num_units=NUM_UNITS,loss=LOSS,learning_rate=LEARNING_RATE):
    #generate train sequences
    inputs,targets=generate_training_sequences(SEQUENCE_LENGTH)
    #build network
    model=build_model(output_units,num_units,loss,learning_rate)
    tbCallback=TensorBoard(log_dir='logs/{}'.format(NAME))
    #train the model
    model.fit(inputs,targets,epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=[tbCallback])
    #save model
    model.save(SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()