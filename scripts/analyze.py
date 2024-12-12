import matplotlib.pyplot as plt

def analyze_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Dokładność treningu')
    plt.plot(epochs_range, val_acc, label='Dokładność walidacji')
    plt.legend(loc='lower right')
    plt.title('Dokładność')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Strata treningu')
    plt.plot(epochs_range, val_loss, label='Strata walidacji')
    plt.legend(loc='upper right')
    plt.title('Strata')

    plt.show()