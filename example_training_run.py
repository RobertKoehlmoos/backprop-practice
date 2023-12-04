from keras.datasets import mnist
import numpy as np
import time

import neural_network
import neural_network_batch

np.random.seed(42)  # good science is reproducible

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print(f'X_train: {train_X.shape}')
print(f'Y_train: {train_y.shape}')
print(f'X_test:  {test_X.shape}')
print(f'Y_test:  {test_y.shape}')

# Transforming the keras supplied mnist dataset to work for training the non-batched neural network
train_x_flattened = [x.flatten().reshape((-1, 1))/255 for x in train_X]
test_x_flattened = [x.flatten().reshape((-1, 1))/255 for x in test_X]
train_y_vectorized = [np.zeros((10, 1)) for _ in train_y]
for i, y in enumerate(train_y):
    train_y_vectorized[i][y] = 1.0
test_y_vectorized = [np.zeros((10, 1)) for _ in test_y]
for i, y in enumerate(test_y):
    test_y_vectorized[i][y] = 1.0

my_nn = neural_network.MLP([28*28, 30, 30, 10])
start = time.time()
my_nn.sgd(train_x_flattened, train_y_vectorized, test_x_flattened, test_y_vectorized, epochs=10, batch_size=100)
print(f'Time to train for non-batched implementation: {time.time() - start}')

# for the batched version
train_x_flattened = train_X.reshape(-1, 28*28)/255
test_x_flattened = test_X.reshape(-1, 28*28)/255
train_y_vectorized = np.eye(10)[train_y]
test_y_vectorized = np.eye(10)[test_y]

my_nn = neural_network_batch.MLP([28*28, 30, 30, 10])
start = time.time()
my_nn.sgd(train_x_flattened, train_y_vectorized, test_x_flattened, test_y_vectorized, epochs=10, batch_size=100)
print(f'Time to train for batched implementation: {time.time()- start}')

# Example Output
# X_train: (60000, 28, 28)
# Y_train: (60000,)
# X_test:  (10000, 28, 28)
# Y_test:  (10000,)
# Accuracy for epoch 0: 0.6557
# Accuracy for epoch 1: 0.7209
# Accuracy for epoch 2: 0.7543
# Accuracy for epoch 3: 0.7828
# Accuracy for epoch 4: 0.7904
# Accuracy for epoch 5: 0.7953
# Accuracy for epoch 6: 0.8029
# Accuracy for epoch 7: 0.8106
# Accuracy for epoch 8: 0.8129
# Accuracy for epoch 9: 0.8166
# Time to train for non-batched implementation: 72.24560856819153
# Accuracy for epoch 0: 0.6151
# Accuracy for epoch 1: 0.7226
# Accuracy for epoch 2: 0.7448
# Accuracy for epoch 3: 0.7543
# Accuracy for epoch 4: 0.7592
# Accuracy for epoch 5: 0.7614
# Accuracy for epoch 6: 0.7695
# Accuracy for epoch 7: 0.7724
# Accuracy for epoch 8: 0.7696
# Accuracy for epoch 9: 0.7711
# Time to train for batched implementation: 57.75354361534119
#
# Process finished with exit code 0

