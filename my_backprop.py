from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print(f'X_train: {train_X.shape}')
print(f'Y_train: {train_y.shape}')
print(f'X_test:  {test_X.shape}')
print(f'Y_test:  {test_y.shape}')
