from create_model import create_cnn
from prep import prepare_data
from tensorflow.keras.callbacks import EarlyStopping


def train():
    # Przygotowanie danych
    train_data, val_data = prepare_data()

    # Parametry wejściowe modelu
    input_shape = (128, 128, 3)
    num_classes = len(train_data.class_indices)

    # Tworzenie modelu
    model = create_cnn(input_shape, num_classes)

    # Early stopping dla uniknięcia przeuczenia
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Trenowanie modelu
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=40,
        steps_per_epoch=len(train_data),
        validation_steps=len(val_data),
        callbacks=[early_stopping],
        verbose=1
    )

    # Zapisanie modelu
    model.save('models/truman_classifier_0.1.keras')
    return history


if __name__ == '__main__':
    from analyze import analyze_training

    # Trenowanie modelu i analiza wyników
    history = train()
    analyze_training(history)