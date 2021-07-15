from tests.test_train import test_train
from tests.test_train import test_test

print('Training the data set....')
train = test_train()
train.run()

print('\n\nDecomposing a test set...')
test = test_test()
test.run()
