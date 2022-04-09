import numpy as np
from PIL import Image
from matplotlib import pyplot as pp

class Net:

    def __init__(self, m_data_features, o_target_labels, hidden_layer_size=20,
                 hidden_layer_activation="relu", regularization_strength=0, learning_rate=0.1,
                 learning_rate_decay=1.1, label_smoothing=1.):

        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = o_target_labels
        self.hidden_layer_activation = hidden_layer_activation
        self.label_smoothing = label_smoothing
        self.alpha = learning_rate
        self.lr_decay = learning_rate_decay
        self.regularization_strength = regularization_strength

        self.generator = np.random.RandomState()
        self.W1 = self.generator.uniform(size=[m_data_features, hidden_layer_size], low=-0.1,
                                         high=0.1)
        self.W2 = self.generator.uniform(size=[hidden_layer_size, o_target_labels], low=-0.1,
                                         high=0.1)
        self.b1 = np.zeros(hidden_layer_size)
        self.b2 = np.zeros(o_target_labels)

        self.debug = False

    def cost(self, y, t):
        return np.log(y[t])

    def forward(self, x, training=False):
        if self.debug:
            print(x)
            print("times")
            print(self.W1)
            print("plus")
            print(self.b1)
        raw_output_after_hidden_layer = x @ self.W1 + self.b1
        if self.hidden_layer_activation == "relu":
            output_after_hidden_layer = np.maximum(raw_output_after_hidden_layer, 0)
        elif self.hidden_layer_activation == "tanh":
            output_after_hidden_layer = np.tanh(raw_output_after_hidden_layer)
        else:
            raise ValueError("unknown activaction function for the hidden layer")
        raw_output_after_output_layer = output_after_hidden_layer @ self.W2 + self.b2
        if self.debug:
            print("equals to")
            print(raw_output_after_hidden_layer)
            print("hidden layer activation", self.hidden_layer_activation)
            print(output_after_hidden_layer)
            print("times")
            print(self.W2)
            print("plus")
            print(self.b2)
            print("equals to")
            print(raw_output_after_output_layer)

        output_layer_activation = "none" if self.output_layer_size == 1 else "softmax"
        if output_layer_activation == "softmax":
            # Note that the `axis=-1, keepdims=True` allow processing both 1D and 2D inputs, always computing the sofmax "on the last dimension".
            # softmax(z) = softmax(z + any_constant) => softmax(z) = softmax(z - maximum_of_z) to avoid owerflowing of softmax

            prediction = np.exp(
                raw_output_after_output_layer - np.max(raw_output_after_output_layer, axis=-1,
                                                       keepdims=True))

            prediction /= np.sum(prediction, axis=-1, keepdims=True)  # Softmax normalization
        else:
            prediction = raw_output_after_output_layer[
                0]  # return number not a list containing one number
        if self.debug:
            print("activation", output_layer_activation)
            print(prediction)
        if training:
            return output_after_hidden_layer, prediction
        else:
            return prediction

    def predict(self, data):
        o = self.forward(data)
        if self.output_layer_size == 0:
            return o
        else:
            return np.argmax(o, axis=-1)

    def update_weights(self, batch_x, batch_t):
        output_layer_activation = "none" if self.output_layer_size == 1 else "softmax"

        hidden_layer, output_layer = self.forward(batch_x, training=True)

        # L=NLL=-log(p)=-log(softmax(y_t)) ; y=raw_output_after_output_layer, y_t is the t-th component of it with t indexing an element corresponding to the golden target
        # dL/db2 = dL/dy * dy/db2 (y and b2 are vectors with self.output_layer_size elements)
        # dLdW2 = dL/dy * dy/dW2
        # dL/dy_i = -dlog [ exp(y_t) / sum_over_j{exp(y_j)} ] / dy_i  = -d [y_t - log(sum_over_j{exp(y_j)})] / dy_i = -1*[i==t] + sum_over_j{exp(y_j)* d sum_over_j{exp(y_j) / dy_i
        # dL/dy_i = -1*[i==t] + sum_over_j{exp(y_j)* exp(y_i) = y - t; where t is the golden target vector (0, 0, ..., 1 //for_golden_class//, ..., 0)
        # [i==t] stands for Kronecker delta
        output_layer_gradient = output_layer - np.eye(self.output_layer_size)[batch_t]  # np.eye = diagonal matrix with ones on the diagonal - np trick to implement golden target vector

        # dy/db2 = 1
        # we process data in batches, so don't forget to estimate expected value of the gradient
        self.b2 -= self.alpha * np.mean(output_layer_gradient, axis=0)
        # dy/dW2 = hidden_layer.transposed()
        self.W2 -= self.alpha * ((hidden_layer.T @ output_layer_gradient / len(
            batch_x)) + self.regularization_strength * self.W2)

        # h=raw_output_after_hidden_layer
        # H=output_after_hidden_layer = hidden_layer
        # dLdb1 = dL/dy * dy/dH * dH/dh * dh/db1
        # dLdW1 = dL/dy * dy/dH * dH/dh * dh/dW1
        # dy/dH = W2.T
        if self.hidden_layer_activation == "relu":
            # dH/dh = 1 if H>0 else 0
            hidden_layer_gradient = output_layer_gradient @ self.W2.T * (hidden_layer > 0)
            # print(output_layer_gradient.shape,self.W2.T.shape, (hidden_layer > 0).shape)
        elif self.hidden_layer_activation == "tanh":
            # dH/dh = 1-tanh()^2 =(1-h^2)
            hidden_layer_gradient = output_layer_gradient @ self.W2.T * (
                    1 - hidden_layer * hidden_layer)

        self.b1 -= self.alpha * np.mean(hidden_layer_gradient, axis=0)
        self.W1 -= self.alpha * ((batch_x.T @ hidden_layer_gradient / len(
            batch_x)) + self.regularization_strength * self.W1)

    def train(self, x, t, epochs, batch_size, x_val=None, t_val=None, print_results=True):
        erange = range(1, epochs + 1)

        train_costs = list()
        test_costs = list()
        train_accs = list()
        test_accs = list()

        for i in erange:
            print("Epoch:", i)
            shuffled_indices = self.generator.permutation(len(x))
            shuffled_data = x[shuffled_indices]
            shuffled_target = t[shuffled_indices]

            for j in range(len(x) // batch_size):
                batch_x = shuffled_data[j * batch_size:(j + 1) * batch_size]
                batch_t = shuffled_target[j * batch_size:(j + 1) * batch_size]
                self.update_weights(batch_x, batch_t)

            self.alpha /= self.lr_decay

            if print_results:
                y = self.predict(x)
                good = 0
                for i in range(len(y)):
                    good += 1 if y[i] == t[i] else 0
                train_acc = good / len(t)

                train_accs.append(train_acc)
                train_costs.append(self.cost(y, t))

                y_val = self.predict(x_val)
                good = 0
                for i in range(len(y_val)):
                    good += 1 if y_val[i] == t_val[i] else 0
                val_acc = good / len(t_val)

                test_accs.append(val_acc)
                test_costs.append(self.cost(y_val, t_val))

                print("\tTrain accuracy: %f\n\tTest accuracy: %f\n\tMean train cost: %f\n\tMean test cost: %f" % (train_acc, val_acc, np.mean(train_costs), np.mean(test_costs)))

        pp.scatter(erange, test_accs, label = "Test accuracy")
        pp.scatter(erange, train_accs, label = "Train accuracy")

        pp.xlabel("Epoch")
        pp.ylabel("Accuracy")
        pp.legend()
        pp.show()


def loadImg(path):
    img = Image.open(path).getdata()
    return np.array(list(map(lambda x : int(bool(x)), img)))


data = np.genfromtxt("/home/jakub/Dokumenty/skola/siete/neural/trainset/data", delimiter="\t")
targets = np.genfromtxt("/home/jakub/Dokumenty/skola/siete/neural/trainset/targets.csv", dtype=int)

testdata = np.genfromtxt("/home/jakub/Dokumenty/skola/siete/neural/trainset/testset", delimiter="\t")
testtarget = np.genfromtxt("/home/jakub/Dokumenty/skola/siete/neural/trainset/testtargets", dtype=int)

net = Net(900, 3, hidden_layer_size=10, hidden_layer_activation="tanh",
          regularization_strength=0, learning_rate=0.02, learning_rate_decay=1.,
          label_smoothing=0.)

net.train(data, targets, 200, 10, x_val=testdata, t_val=testtarget)
