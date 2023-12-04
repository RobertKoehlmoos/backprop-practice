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
my_nn.sgd(train_x_flattened[:10000], train_y_vectorized[:10000], test_x_flattened, test_y_vectorized, epochs=20, batch_size=100)
print(f'Time to train for non-batched implementation: {time.time() - start}')

# for the batched version

train_x_flattened = train_X.reshape(-1, 28*28)/255
test_x_flattened = test_X.reshape(-1, 28*28)/255
train_y_vectorized = np.eye(10)[train_y]
test_y_vectorized = np.eye(10)[test_y]

my_nn = neural_network_batch.MLP([28*28, 30, 30, 10])
start = time.time()
my_nn.sgd(train_x_flattened[:10000], train_y_vectorized[:10000], test_x_flattened, test_y_vectorized, epochs=20, batch_size=100)
print(f'Time to train for batched implementation: {time.time()- start}')

# Example Output
# Using TensorFlow backend.
# X_train: (60000, 28, 28)
# Y_train: (60000,)
# X_test:  (10000, 28, 28)
# Y_test:  (10000,)
# Accuracy for epoch 0: 0.3397
# Accuracy for epoch 1: 0.4623
# Accuracy for epoch 2: 0.5375
# Accuracy for epoch 3: 0.585
# Accuracy for epoch 4: 0.6207
# Accuracy for epoch 5: 0.6483
# Accuracy for epoch 6: 0.668
# Accuracy for epoch 7: 0.6832
# Accuracy for epoch 8: 0.6949
# Accuracy for epoch 9: 0.704
# Accuracy for epoch 10: 0.7118
# Accuracy for epoch 11: 0.718
# Accuracy for epoch 12: 0.7247
# Accuracy for epoch 13: 0.7297
# Accuracy for epoch 14: 0.7351
# Accuracy for epoch 15: 0.7418
# Accuracy for epoch 16: 0.7451
# Accuracy for epoch 17: 0.75
# Accuracy for epoch 18: 0.7566
# Accuracy for epoch 19: 0.7613
# Time to train for non-batched implementation: 27.25707173347473
# Accuracy for epoch 0: 0.2284
# Accuracy for epoch 1: 0.34
# Accuracy for epoch 2: 0.4336
# Accuracy for epoch 3: 0.5093
# Accuracy for epoch 4: 0.5689
# Accuracy for epoch 5: 0.6164
# Accuracy for epoch 6: 0.6495
# Accuracy for epoch 7: 0.6767
# Accuracy for epoch 8: 0.6939
# Accuracy for epoch 9: 0.7065
# Accuracy for epoch 10: 0.7115
# Accuracy for epoch 11: 0.7187
# Accuracy for epoch 12: 0.7214
# Accuracy for epoch 13: 0.7268
# Accuracy for epoch 14: 0.7294
# Accuracy for epoch 15: 0.7352
# Accuracy for epoch 16: 0.7404
# Accuracy for epoch 17: 0.7428
# Accuracy for epoch 18: 0.7444
# Accuracy for epoch 19: 0.7461
# Time to train for batched implementation: 19.2605619430542
#
# Process finished with exit code 0
