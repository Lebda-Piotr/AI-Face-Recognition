from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2

def create_cnn(input_shape, num_classes):
    model = Sequential([
        
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        Flatten(),

        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5),

        Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model