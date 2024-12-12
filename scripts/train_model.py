from create_model import create_cnn
from prep import prepare_data

def train():
    train_data, val_data = prepare_data()

    input_shape = (128, 128, 3)
    num_classes = len(train_data.class_indices)
    model = create_cnn(input_shape, num_classes)

    # Trenowanie modelu
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=20,
        steps_per_epoch=len(train_data),
        validation_steps=len(val_data)
    )

    # Zapisanie modelu
    model.save('models/truman_classifier.h5')
    return history

if __name__ == '__main__':
    from analyze import analyze_training

    history = train()
    analyze_training(history)
