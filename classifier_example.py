import tensorslow as ts
import numpy as np

# Define constants
INPUT_H = 28
INPUT_W = 28
INPUT_SHAPE = (INPUT_H * INPUT_W, 1)
N_CLASSES = 10
STEP_SIZE = 0.0001
BATCH_SIZE = 16
MOMENTUM = 0.95
NUM_EPOCHS = 4


def create_dense_layer(in_node, out_num_units, add_activation=True):
    """
    Creates a dense layer: relu(weight_matrix * in_node + bias_vector)
    in_node: ts.Tensor object with shape (N, 1).
    out_num_units: The number of hidden units in the layer.
    add_activation: If a relu activation should be applied.
    """ 
    (in_num_units, _) = in_node.shape

    # Create weight matrix and bias vector
    weight_high = np.sqrt(2 / (in_num_units + out_num_units))
    weights = ts.Variable(np.random.uniform(
        low=-weight_high, high=weight_high, size=(out_num_units, in_num_units)
    ))
    bias = ts.Variable(np.zeros(shape=(out_num_units, 1)) + 0.3)

    # Create operations
    matmul = ts.MatMul(weights, in_node)
    with_bias = ts.Add2(matmul, bias)
    if add_activation:
        return ts.Relu(with_bias)
    else:
        return with_bias


class Model(ts.ComputeGraph):
    def __init__(self):
        # Define ts.Input nodes for the image and ground-truth-label
        self.in_image = ts.Input(INPUT_SHAPE)
        self.in_label = ts.Input((N_CLASSES, ))

        # A sequence of three fully connected layers
        self.dense1 = create_dense_layer(self.in_image, 300)
        self.dense2 = create_dense_layer(self.dense1, 300)
        self.output = create_dense_layer(self.dense2, N_CLASSES, add_activation=False)

        # Transform 10x1 matrix into a 10 element vector
        self.output_flattened = ts.Squeeze(self.output, axis=None)

        # Terminate graph with loss function
        self.loss = ts.SoftMaxCrossEntropy(self.output_flattened, self.in_label)

    def train(self, num_epochs, train_ds, test_ds):
        # Transform training set to [{ts.Input: np.array}] format
        train_ds = [
            {
                self.in_image: image,
                self.in_label: one_hot
            }
            for image, one_hot in train_ds
        ]

        # Run epochs
        for epoch in range(num_epochs):
            print(f"Epoch {epoch}")
            self.print_confusion_matrix(test_ds)
            loss = self.momentum_sgd_epoch(STEP_SIZE, self.loss, BATCH_SIZE, MOMENTUM, train_ds)

    def classify(self, x):
        """Returns the predicted label of x"""
        output = self.output_flattened.evaluate({self.in_image: x})
        return np.argmax(output)

    def print_confusion_matrix(self, test_ds):
        matrix = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for image, one_hot_truth in test_ds:
            truth = np.argmax(one_hot_truth)
            prediciton = self.classify(image)
            matrix[truth, prediciton] += 1

        print("Accuracy:", np.trace(matrix) / np.sum(matrix))
        print("Confusion matrix: \n", matrix)


def mnist_file_parse(filename):
    result=np.loadtxt(filename, delimiter=",")
    # result /= 255
    dataset = []
    for row in result:
        label = int(row[0])
        one_hot = np.zeros((N_CLASSES, ), dtype=float)
        one_hot[label] = 1.0
        datapoint = row[1:]
        datapoint = np.reshape(datapoint, INPUT_SHAPE)
        datapoint /= 255
        dataset.append((datapoint, one_hot))

    return dataset


def load_mnist():
    return (
        mnist_file_parse('data/mnist_train.csv'),
        mnist_file_parse('data/mnist_test.csv')
    )


model = Model()
print("Loading dataset...")
train_ds, test_ds = load_mnist()
model.train(NUM_EPOCHS, train_ds, test_ds)