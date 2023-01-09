import os
import pickle
import numpy as np



def load_CIFAR_10_batch(filename):
    """
    Load a single batch of CIFAR-10.
    Returns a tuple with samples and corresponding labels.
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file, encoding='latin1')

        X = data['data']
        y = data['labels']

        return X.reshape(10000, 3, 32, 32).astype(np.float32), np.array(y)



def load_CIFAR_10(root):
    """
    Load the complete CIFAR-10 data set.
    Returns a tuple with training samples and labels and test samples and labels.
    """
    Xs = []
    ys = []

    for batch in range(1, 6):
        filename = os.path.join(root, f'data_batch_{batch}')

        X, y = load_CIFAR_10_batch(filename)

        Xs.append(X)
        ys.append(y)    

    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)

    del X, y

    X_test, y_test = load_CIFAR_10_batch(os.path.join(root, 'test_batch'))

    return X_train, y_train, X_test, y_test



def get_CIFAR_10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load and preprocess the CIFAR-10 data set.
    Partitions the data in training, validation, test and development set.
    """
    X_train, y_train, X_test, y_test = load_CIFAR_10('datasets/cifar-10-batches-py')

    indices = np.arange(num_training, 50000)
    X_val = X_train[indices]
    y_val = y_train[indices]

    indices = np.arange(num_training)
    X_train = X_train[indices]
    y_train = y_train[indices]

    indices = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[indices]
    y_dev = y_train[indices]

    indices = np.arange(num_test)
    X_test = X_test[indices]
    y_test = y_test[indices]

    mean = np.mean(X_train, axis=0)

    X_train -= mean
    X_val   -= mean
    X_test  -= mean
    X_dev   -= mean
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'X_val': X_val,
        'y_val': y_val,
        'X_dev': X_dev,
        'y_dev': y_dev
    }
