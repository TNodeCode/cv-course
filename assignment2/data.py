import os
import pickle
import numpy as np



def load_CIFAR_10_batch(filename):
    """
    Load a single batch of CIFAR-10.
    Returns a tuple with samples and corresponding labels.

    Parameters:
        - filename: File name of the stored batch.

    Returns:
        - CIFAR-10 batch with image channels last, converted to float.

    """
    with open(filename, 'rb') as file:
        data = pickle.load(file, encoding='latin1')

        X = data['data']
        Y = data['labels']
        
    return X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32), np.array(Y)



def load_CIFAR_10(root):
    """
    Load the complete CIFAR-10 data set.

    Parameters:
        - root: Directory where dataset is stored.

    Returns:
        - X_train: Training images.
        - y_train: Training labels.
        - X_test: Test images.
        - y_test: Test labels.

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


