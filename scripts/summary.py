from tensorflow.keras.models import load_model

model_path = 'models/truman_classifier_0.2.h5'

model = load_model(model_path)

model.summary()
