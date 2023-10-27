#Daniel Tran
#CS445 Prog HW1
import matplotlib.pyplot as plt
import numpy as np
import pickle
from decimal import *
from scipy.special import expit as sigmoid

#class for getting the data
class Data_mnist(object):
    #load data file
    def __init__(self, filename='norm_mnist.dat'):
        r_ts, r_xs, t_ts, t_xs = self.__load_data(filename)
        self.train_ts = r_ts
        self.train_xs = r_xs
        self.test_ts = t_ts
        self.test_xs = t_xs
    #load the .csv files
    def __load_data(self, filename):
        #load training file
        trainf = np.loadtxt('mnist_train.csv', delimiter=',', unpack=False)
        #load testing file
        testf = np.loadtxt('mnist_test.csv', delimiter=',', unpack=False)
        #separate values from samples
        t_train = trainf[:, 0]
        t_train = t_train.astype(int)
        t_test = testf[:, 0]
        t_test = t_test.astype(int)
        #normalize data
        trainf = trainf / 255.0
        trainf[:, 0] = 1
        testf = testf / 255.0
        testf[:, 0] = 1
        #convert data to binary
        with open(filename, 'wb') as f:
            pickle.dump([t_train, trainf, t_test, testf], f)
        return t_train, trainf, t_test, testf

#Neural Network Class
class NNetwork(object):
    #Constructor
    def __init__(self, num_input, num_hidden, num_output, NN_rate = 0.01, NN_momentum=0.9,
                 target=0.9, num_epoch=50):
        self.NN_input = num_input
        self.NN_hidden = num_hidden
        self.NN_output = num_output
        self.NN_epoch = num_epoch
        self.NN_rate = NN_rate
        self.NN_momentum = NN_momentum

        # Initialize weights
        self.weight1, self.weight2 = self.__init_weights()
        #Storage for previous weights
        self.w1 = np.zeros(self.weight1.shape)
        self.w2 = np.zeros(self.weight2.shape)
        #calculate the errors with target
        self.target = target
        #create x and y variables for plot
        self.x_r = [i for i in range(num_epoch + 1)]
        self.y_r = [0]*(num_epoch + 1)
        self.x_t = [i for i in range(num_epoch + 1)]
        self.y_t = [0]*(num_epoch + 1)

    #inital weight function
    def __init_weights(self):
        #randomize postive and negative weights
        num_weights = self.NN_input * self.NN_hidden
        weight1 = np.random.uniform(-0.05, 0.05, num_weights)
        weight1 = weight1.reshape(self.NN_input, self.NN_hidden)

        num_weights = (self.NN_hidden + 1) * self.NN_output
        weight2 = np.random.uniform(-0.05, 0.05, num_weights)
        weight2 = weight2.reshape(self.NN_hidden + 1, self.NN_output)
        return weight1, weight2

    #function to update weights
    def __update_weights(self, x_i, error1, add_col, error2):
        #comupte first layer
        delta_weight1 = (self.NN_rate * np.outer(x_i, error1[1:])) + (self.NN_momentum * self.w1)
        #update weights
        self.weight1 += delta_weight1
        #save the current delta
        self.w1 = delta_weight1
        # Comupte second layer
        delta_weight2 = (self.NN_rate * np.outer(add_col, error2)) + (self.NN_momentum * self.w2)
        #update weights
        self.weight2 += delta_weight2
        # Save the current delta
        self.w2 = delta_weight2
        return

    #training functions
    def train(self, x, t, num_sample1, X, T, num_sample2):
        #test the training and test data and print results
        print('Epoch: 0\nTraining', end=' ')
        self.y_r[0] = self.test(x, t, num_sample1)
        print('Test', end=' ')
        self.y_t[0] = self.test(X, T, num_sample2)

        #add space for the hidden nodes
        add_col = np.ones(self.NN_hidden+1)
        #reformat array of 10 x number of samples
        target_mat = np.ones((self.NN_output, self.NN_output), float) - self.target
        np.fill_diagonal(target_mat, self.target)

        #training algorithm for sixe of epoch
        for epoch in range(self.NN_epoch):
            for i in range(num_sample1):
                add_col, out_k = self.__forward(x[i, :], add_col)
                #calculate errors
                error_o = out_k * (1 - out_k) * (target_mat[t[i]] - out_k)
                error_h = add_col * (1 - add_col) * np.dot(self.weight2, error_o)
                #update weights
                self.__update_weights(x[i, :], error_h, add_col, error_o)

            #after each epoch, training/test accuracy will be tested and printed
            print('\nEpoch: ' + str(epoch+1))
            print('Training', end=' ')
            self.y_r[epoch+1] = self.test(x, t, num_sample1)
            print('Test', end=' ')
            self.y_t[epoch+1] = self.test(X, T, num_sample2)
        #generate confusion matrix after last epoch
        self.confusion_mat = self.__get_confusion_mat(X, T, num_sample2)
        return

    #function to test and calculate nerual network accuracy
    def test(self, x, t, num_sample1):
        add_col = np.ones(self.NN_hidden+1)
        n_correct = 0
        for i in range(num_sample1):
            _, out_k = self.__forward(x[i, :], add_col)
            if t[i] == np.argmax(out_k):
                n_correct += 1
        #calculate and print accuracy
        accuracy = 100.0 * n_correct / num_sample1
        print('Accuracy = ' + str(accuracy) + '%')
        return accuracy

    #function to forward inputs
    def __forward(self, x, h):
        h[1:] = sigmoid(x.dot(self.weight1))
        o = sigmoid(h.dot(self.weight2))
        return h, o

    #function to get confusion matrix
    def __get_confusion_mat(self, x, t, num_sample1):
        add_col = np.ones(self.NN_hidden+1)
        confusion_mat = np.zeros((10, 10), int)
        for i in range(num_sample1):
            _, out_k = self.__forward(x[i, :], add_col)
            a = t[i]
            b = np.argmax(out_k)
            confusion_mat[a, b] += 1
        return confusion_mat

#main function
if __name__ == "__main__":
    getcontext().prec = 3

    #get mnist data
    data = Data_mnist()
    train_sample, NN_input = data.train_xs.shape
    test_sample, _ = data.test_xs.shape

    #experiment 1
    epochs = 50
    hidden_nodes = [20, 50, 100]
    momentum = 0.9
    NN_rate = 0.1
    for nodes in hidden_nodes:
        #initialize neural net
        nn = NNetwork(num_input=NN_input,
                       num_hidden=nodes,
                       num_output=10,
                       NN_rate=NN_rate,
                       NN_momentum=momentum,
                       target=0.9,
                       num_epoch=epochs)

        #call training function for neural network
        nn.train(data.train_xs, data.train_ts, train_sample,
                 data.test_xs, data.test_ts, test_sample)

        #generate plot for experiment 1 when changing hidden node values
        plt.plot(nn.x_r, nn.y_r, label='Training Data')
        plt.plot(nn.x_t, nn.y_t, label='Test Data')
        plt.ylim([0, 100])
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Experiment 1: All Training Samples' +
                  '\nHidden Layers = ' + str(nodes) +', Momentum = ' + str(momentum))
        plt.ylim([80, 100])
        save_file = ('experiment1layer' + str(nodes) +'.png')
        plt.savefig(save_file)
        plt.clf()

        print(save_file)
        print(nn.confusion_mat)

    #experiment 2
    epochs = 50
    hidden_nodes = 100
    momentum = [0, 0.25, 0.50]
    NN_rate = 0.1
    for m in momentum:
        nn = NNetwork(num_input=NN_input, 
                       num_hidden=hidden_nodes,
                       num_output=10,
                       NN_rate=NN_rate,
                       NN_momentum=m,
                       target=0.9,
                       num_epoch=epochs)

        nn.train(data.train_xs, data.train_ts, train_sample,
                 data.test_xs, data.test_ts, test_sample)

        #generate plot for experiment 2 when changing momentum values
        plt.plot(nn.x_r, nn.y_r, label='Training Data')
        plt.plot(nn.x_t, nn.y_t, label='Test Data')
        plt.ylim([0, 100])
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Experiment 2: All Training Samples' +
                  '\nHidden Layers = ' + str(nodes) +', Momentum = ' + str(momentum))
        plt.ylim([80, 100])
        save_file = ('experiment2momentum' + str(momentum) +'.png')
        plt.savefig(save_file)
        plt.clf()

        print(save_file)
        print(nn.confusion_mat)

    #experiment 3
    #dividing the training sample by 4
    batch_size25 = int(train_sample/4)
    epochs = 50
    hidden_nodes = 100
    momentum = 0.9
    NN_rate = 0.1
    nn = NNetwork(num_input=NN_input,
                   num_hidden=nodes,
                   num_output=10,
                   NN_rate=NN_rate,
                   NN_momentum=momentum,
                   target=0.9,
                   num_epoch=epochs)

    nn.train(data.train_xs[0:batch_size25], data.train_ts[0:batch_size25],
             batch_size25, data.test_xs, data.test_ts, test_sample)

    #generates the plot for exerpiment 3 when using one quarter training samples
    plt.plot(nn.x_r, nn.y_r, label='Training Data')
    plt.plot(nn.x_t, nn.y_t, label='Test Data')
    plt.ylim([0, 100])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Experiment 3: One Quarter Training Samples' +
                '\nHidden Layers = ' + str(nodes) +', Momentum = ' + str(momentum))
    plt.ylim([80, 100])
    save_file = ('experiment3quarter.png')
    plt.savefig(save_file)
    plt.clf()

    print(save_file)
    print(nn.confusion_mat)

    #dividing the training sample in halves
    batch_size50 = int(train_sample/2)
    nn = NNetwork(num_input=NN_input,
                   num_hidden=nodes,
                   num_output=10,
                   NN_rate=NN_rate,
                   NN_momentum=momentum,
                   target=0.9,
                   num_epoch=epochs)

    nn.train(data.train_xs[20000:20000+batch_size50],
             data.train_ts[20000:20000+batch_size50],
             batch_size50,
             data.test_xs,
             data.test_ts,
             test_sample)

    #generates the plot for experiment 3 when using one half training samples
    plt.plot(nn.x_r, nn.y_r, label='Training Data')
    plt.plot(nn.x_t, nn.y_t, label='Test Data')
    plt.ylim([0, 100])
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Experiment 3: One Half Training Samples' +
                '\nHidden Layers = ' + str(nodes) +', Momentum = ' + str(momentum))
    plt.ylim([80, 100])
    save_file = ('experiment3half.png')
    plt.savefig(save_file)
    plt.clf()

    print(save_file)
    print(nn.confusion_mat)