from keras.datasets import mnist
import numpy as np

import neural_network
(train_X, train_y), (test_X, test_y) = mnist.load_data()

print(f'X_train: {train_X.shape}')
print(f'Y_train: {train_y.shape}')
print(f'X_test:  {test_X.shape}')
print(f'Y_test:  {test_y.shape}')

# Transforming the keras supplied mnist dataset to work for training the neural network
train_x_flattened = [x.flatten().reshape((-1, 1)) for x in train_X]
test_x_flattened = [x.flatten().reshape((-1, 1)) for x in test_X]
train_y_vectorized = [np.zeros((10, 1)) for _ in train_y]
for i, y in enumerate(train_y):
    train_y_vectorized[i][y] = 1.0
test_y_vectorized = [np.zeros((10, 1)) for _ in test_y]
for i, y in enumerate(test_y):
    test_y_vectorized[i][y] = 1.0

my_nn = neural_network.MLP([28*28, 30, 30, 10])

my_nn.sgd(train_x_flattened[:10000], train_y_vectorized[:10000], test_x_flattened, test_y_vectorized, epochs=20, batch_size=100)

# example training run
# Accuracy for epoch 0: 0.4613
# Accuracy for epoch 1: 0.5337
# Accuracy for epoch 2: 0.5739
# Accuracy for epoch 3: 0.6145
# Accuracy for epoch 4: 0.6371
# Accuracy for epoch 5: 0.663
# Accuracy for epoch 6: 0.6785
# Accuracy for epoch 7: 0.6952
# Accuracy for epoch 8: 0.7015
# Accuracy for epoch 9: 0.7107
# Accuracy for epoch 10: 0.7175
# Accuracy for epoch 11: 0.7239
# Accuracy for epoch 12: 0.7302
# Accuracy for epoch 13: 0.7332
# Accuracy for epoch 14: 0.7388
# Accuracy for epoch 15: 0.7427
# Accuracy for epoch 16: 0.7479
# Accuracy for epoch 17: 0.7464
# Accuracy for epoch 18: 0.7513
# Accuracy for epoch 19: 0.7589
#
# Process finished with exit code 0
