import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1" 

import numpy as np
import csv
import time
import matplotlib.pyplot as plt


# Data Loading and Preprocessing
def load_data(data_path, labels_path=None):
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        data = np.array(list(reader))

    if labels_path:
        with open(labels_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            labels = np.array([int(row[0]) for row in reader])
            labels = labels.reshape(1, -1)
        
            max_label = np.max(labels)
            min_label = np.min(labels)
            labels = (labels - min_label) / (max_label - min_label)
        return data, labels, min_label, max_label
    else:
        return data

def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / std

def create_category_mappings(train_data, test_data, categorical_columns):
    category_mappings = {}
    for col_index in categorical_columns:
        combined = np.concatenate((train_data[:, col_index], test_data[:, col_index]))
        category_mappings[col_index] = np.unique(combined)
    return category_mappings

def one_hot_encode(features, all_categories):
    category_index = {category: idx for idx, category in enumerate(all_categories)}
    encoded = np.zeros((len(features), len(all_categories)))
    for i, feature in enumerate(features):
        if feature in category_index:
            encoded[i, category_index[feature]] = 1
    return encoded

def preprocess_data(X, numeric_columns, categorical_columns, category_mappings):
    numeric_data = X[:, numeric_columns].astype(float)
    normalized_data = normalize_features(numeric_data)
    
    one_hot_encoded_data = []
    for i, col_index in enumerate(categorical_columns):
        all_categories = category_mappings[col_index]
        encoded_data = one_hot_encode(X[:, col_index], all_categories)
        one_hot_encoded_data.append(encoded_data)
    
    preprocessed_data = np.hstack([normalized_data] + one_hot_encoded_data)
    return preprocessed_data

# Neural Network Initialization
def initialize_parameters(layer_dims):
    np.random.seed(83)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

# Forward Propagation
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache

def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, activation_cache = relu(Z)
    else:
        A = Z
        activation_cache = Z
    cache = (linear_cache, activation_cache)
    return A, cache

def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='relu')
    caches.append(cache)
    return AL, caches


# Cost Function
def compute_cost(AL, Y, parameters, lambda_=0.01):
    m = Y.shape[1]
    regularized_sum = sum(np.sum(np.square(parameters['W' + str(l)])) for l in range(1, len(parameters)//2+1))
    cross_entropy_cost = (1 / (2 * m)) * np.sum(np.square(AL - Y))
    regularization_cost = (lambda_ / (2 * m)) * regularized_sum
    cost = cross_entropy_cost + regularization_cost
    return cost


# Backward Propagation
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = 1 / m * np.dot(dZ, A_prev.T)
    db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    else:
        dZ = dA
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def backward_propagation(AL, Y, caches, parameters, lambda_=0.01):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    epsilon = 1e-8
    dAL = - (np.divide(Y, AL + epsilon) - np.divide(1 - Y, 1 - AL + epsilon))
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation=None)

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
        dW_temp += (lambda_ / m) * parameters["W" + str(l + 1)]
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def clip_gradients(grads, maxValue):
    for key in grads.keys():
        np.clip(grads[key], -maxValue, maxValue, out=grads[key])
    return grads


# Update Parameters
def update_parameters(parameters, grads, learning_rate):
    """Update parameters using gradient descent."""
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
    return parameters


# Predictions
def predict(X, parameters):
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, _ = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')
    AL, _ = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation=None)
    return AL

def model(X, Y, layers_dims, learning_rate, num_epochs, batch_size, print_cost, lambda_=0.01):
    np.random.seed(83)
    costs = []
    parameters = initialize_parameters(layers_dims)
    m = X.shape[1]
    Y = Y.reshape(1, -1)

    for i in range(num_epochs):
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]
        
        for j in range(0, m, batch_size):
            X_batch = X_shuffled[:, j:j+batch_size]
            Y_batch = Y_shuffled[:, j:j+batch_size]
            
            AL, caches = forward_propagation(X_batch, parameters)
            cost = compute_cost(AL, Y_batch, parameters, lambda_)
            grads = backward_propagation(AL, Y_batch, caches, parameters, lambda_)

            maxValue = 5
            grads = clip_gradients(grads, maxValue)
            parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


def save_predictions(predictions, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['BEDS'])
        for pred in np.nditer(predictions):
            writer.writerow([int(pred)])


def main():
    start_time = time.time() 
    train_X, train_Y, min_label, max_label = load_data('train_data.csv', 'train_label.csv')
    test_X = load_data('test_data.csv')

    numeric_columns = [2, 3, 4, 14, 15]  # PRICE, BATH, PROPERTYSQFT, LATITUDE, LONGITUDE
    categorical_columns = [1, 6, 8, 9, 10] # TYPE, STATE, ADMINISTRATIVE_AREA_LEVEL_2, LOCALITY, SUBLOCALITY

    # Preprocess the data
    train_category_mappings = create_category_mappings(train_X, test_X, categorical_columns)
    train_X_preprocessed = preprocess_data(train_X, numeric_columns, categorical_columns, train_category_mappings)
    test_X_preprocessed = preprocess_data(test_X, numeric_columns, categorical_columns, train_category_mappings)

    # Initialize the network architecture
    n_x = train_X_preprocessed.shape[1]
    n_h = [32, 16]
    n_y = 1 
    layers_dims = [n_x] + n_h + [n_y]

    # Train the model
    parameters = model(train_X_preprocessed.T, train_Y.T, layers_dims, learning_rate=0.001, num_epochs=1000, batch_size=16, print_cost=True, lambda_=0.01)
    predictions = predict(test_X_preprocessed.T, parameters)
    predictions_rescaled = predictions * (max_label - min_label) + min_label
    save_predictions(predictions_rescaled, "output.csv")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total runtime: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()