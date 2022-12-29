import tensorflow as tf
import joblib

from modules.models.model import Model #
from modules.data.data_preprocess import preprocess #
from modules.evaluation.evaluation import model_evaluation #
from modules.visualization.vistualization import Vistualization #

#loading data and preprocess
print("Loading data....!")
train_set, test_set = preprocess.train_preprocess()

#Data_vistualization.data_visutalization(train_set,test_set)

print("Loading Model....!")
model_call = Model.cnnModel() #

model_call.compile(
  optimizer='adam',
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  metrics=['accuracy'])


model_call.fit(
  train_set, 
  validation_data=test_set, 
  epochs=1,
  shuffle = True,
  batch_size = 32
)


#save the model
model_name = "trained_models/trained_model_01"
with open(model_name, "wb") as file:
    joblib.dump(value=model_call, filename=model_name)


# #load model
# model_name = "trained_models/trained_model_00"
# loaded_model = joblib.load(model_name)

model_evaluation.evaluation(model_call,train_set,test_set)
