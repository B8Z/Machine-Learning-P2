import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import data_generator

# implement a function modify the data with a polynomial basis function
def poly_basis(X, degree):
    '''
    Applies a polynomial basis function of variable degree to data X
    :param X: data to apply basis function to
    :param degree: degree of the polynomial of the basis function
    :return: the data after the basis function is applied
    '''
    deg = int(degree)
    count = 2
    new_data = X
    if(deg == 1):
        return new_data
    while count <= deg:
        X_copy = X ** count
        new_data = np.concatenate([new_data, X_copy], axis=1)
        count += 1
    return new_data


# A function to add a bias as the first dimension of the data
def add_bias(X):
    '''
    Adds a bias to the data X
    :param X: the data
    :return: the data with bias added
    '''
    bias = np.ones((len(X), 1), dtype=float)
    new_data = np.append(bias, X, axis=1)
    return new_data


# implement the closed form solution of linear regression
# implement the closed form solution of linear regression with L2 regularization
def train_regression_analytical(X, Y, penalty=0, regularization='none'):
    '''
    Analytical solution to linear regression

    :param X: The data
    :param Y: The labels
    :param penalty: regularization penalty
    :param regularization: the regularization type (either 'none' or 'l2')
    :return: theta - the model weights
    '''
    transpose_x = np.transpose(X)
    XTX = np.matmul(transpose_x, X)
    if regularization == 'none':
        inv_XTX = np.linalg.inv(XTX)
        inv_XTX_XT = np.matmul(inv_XTX, transpose_x)
        theta = np.matmul(inv_XTX_XT, Y)
        return theta
    elif regularization == 'l2':
        identity_matrix = np.identity(len(XTX))
        identity_matrix_penalty = penalty * identity_matrix
        XTX_I = np.add(XTX, identity_matrix_penalty)
        inv_XTX_I = np.linalg.inv(XTX_I)
        inv_XTX_I_XT = np.matmul(inv_XTX_I, transpose_x)
        theta = np.matmul(inv_XTX_I_XT, Y)
        return theta


# implement gradient descent (basic)
# implement gradient descent for logistic regression log_on=True
# implement gradient descent with L1 and L2 regularization
def train_gradient_descent(X, Y, max_epoch, learning_rate, log_on=False, penalty=0, regularization='none'):
    '''
    Gradient descent solution to linear and logistic regression
    :param X: the data
    :param Y: the labels
    :param max_epoch: the maximum epochs to iterate
    :param learning_rate: the learning rate during updates
    :param log_on: True or False indicating whether to perform logistic regression
    :param penalty: the regularization penalty
    :param regularization: the regularization type (either 'none', 'l1', or 'l2')
    :return: theta - the model weights
    '''
    transpose_x = np.transpose(X)
    XTX = np.matmul(transpose_x, X)
    theta = np.random.randn(len(X[0]), 1)
    if log_on or regularization == 'none':
        for _ in range(max_epoch):
            yhat = predict(theta, X, log_on)
            loss = yhat - Y
            gradient = np.dot(transpose_x, loss) / len(X)
            theta = theta - learning_rate * gradient
        return theta
    elif regularization == 'l1':
        for _ in range(max_epoch):
            yhat = predict(theta, X, log_on)
            loss = yhat - Y
            gradient = np.dot(transpose_x, loss) / len(X)
            l1_component = penalty * np.sign(theta)
            theta = theta - ((learning_rate * gradient) + l1_component)
        return theta
    elif regularization == 'l2':
        for _ in range(max_epoch):
            yhat = predict(theta, X, log_on)
            loss = yhat - Y
            gradient = np.dot(transpose_x, loss) / len(X)
            l2_component = penalty * theta
            theta = theta - ((learning_rate * gradient) + l2_component)
        return theta

def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

# implement a function calculate the predicted values based on data and model weights.
# for logistic regression add a logistic component (log_on = True = predict with a logistic function)
def predict(theta, X, log_on=False):
    '''
    Predicts new values for input data (X) using model weights (theta)
    :param theta: the model weights
    :param X: the input data
    :param log_on: True or False indicating whether to predict with a logarithmic function
    :return: the predictions for input data
    '''
    if log_on:
        scores = np.dot(X, theta)
        predictions = sigmoid(scores)
        return predictions
    else:
        scores = np.dot(X, theta)
        return scores


# Implement Z-scale normalization
def normalize(X, mean=None, stdev=None):
    '''
    normalizes the data by Z-scaling (mean=0, stdev=1). If mean or standard deviation are specified, normalize using
    the input values. Otherwise, calculate the mean and std dev from the data
    :param X: the data
    :param mean: a previously calculated mean
    :param stdev: a previously calculated standard deviation
    :return: a 3-tuple consisting of the normalized data, the mean, and the stdev (data, mean, stdev)
    '''
    if stdev is None and mean is None:
        means = [0 for i in range(len(X[0]))]
        for i in range(len(X[0])):
            column_values = [row[i] for row in X]
            means[i] = sum(column_values) / float(len(X))
        stdevs = [0 for i in range(len(X[0]))]
        for i in range(len(X[0])):
            variance = [pow(row[i] - means[i], 2) for row in X]
            stdevs[i] = sum(variance)
        stdevs = [np.sqrt(j / (float(len(X) - 1))) for j in stdevs]
        standardized_data = (X - means) / (stdevs)
        tuple_values = (standardized_data, means, stdevs)
    elif mean is None:
        means = [0 for i in range(len(X[0]))]
        for i in range(len(X[0])):
            column_values = [row[i] for row in X]
            means[i] = sum(column_values) / float(len(X))
        standardized_data = (X - means) / (stdev)
        tuple_values = (standardized_data, means, stdev)
    elif stdev is None:
        stdevs = [0 for i in range(len(X[0]))]
        for i in range(len(X[0])):
            variance = [pow(row[i] - mean[i], 2) for row in X]
            stdevs[i] = sum(variance)
        stdevs = [np.sqrt(x / (float(len(X) - 1))) for x in stdevs]
        standardized_data = (X - mean) / (stdevs)
        tuple_values = (standardized_data, mean, stdevs)
    else:
        standardized_data = (X - mean) / (stdev)
        tuple_values = (standardized_data, mean, stdev)
    return tuple_values


# implement a function to calculate loss. The two loss functions (objective functions) are Normnalized Binary
# Cross Entropy (BCE) and Mean Squared Error (MSE)
# as you get further along in your experiments, don't forget to add complexity penalties (from regularization) to
# your loss calculation
def calculate_loss(X, Y, theta, loss_function, log_on=False, regularization_type='none', penalty=0):
    '''
    Calculates the loss of the model given the data and labels
    :param X: the data
    :param Y: the labels
    :param theta: the model weights
    :param loss_function: string specifying the loss function ('BCE' or 'MSE')
    :param log_on: True or False indicating whether to predict with a logarithmic function
    :param regularization_type: the regularization type (either 'none', 'l1', or 'l2')
    :param penalty: the regularization penalty
    :return: the loss
    '''

    if (loss_function == "BCE"):
        yhat = predict(theta, X, log_on)
        yi_log_yhat = np.matmul(Y.T, np.log(yhat))
        log_yhat = np.log(1 - yhat)
        second_piece = np.matmul((1 - Y), log_yhat.T)
        both_parts = yi_log_yhat + second_piece
        bce = -np.sum(both_parts)
        return bce
    elif (loss_function == "MSE"):
        # add the regularization component
        if (regularization_type == 'l1'):#adding regularization to whatever comes out of loss_function (least squares) #penalty * complexity
            yhat = predict(theta, X, log_on)
            loss = yhat - Y
            cost = np.sum(loss ** 2) / (2 * len(X))
            l1 = cost + penalty * np.sum(np.abs(theta))# * Complexity -> CALCULATE sum of |theta|
            return l1
        elif (regularization_type == 'l2'):
            yhat = predict(theta, X, log_on)
            loss = yhat - Y
            cost = np.sum(loss ** 2) / (2 * len(X))
            l2 = cost + penalty * np.matmul(theta.T, theta) #sum of ^2 theta values
            return l2
        elif (regularization_type == 'none'):
            yhat = predict(theta, X, log_on)
            loss = yhat - Y
            cost = np.sum(loss ** 2) / (2 * len(X))
            mse = cost
            return mse  # do nothing
        else:
            print("ERROR: unknown regularizer")
            exit()
    else:
        print("ERROR: unknown loss function: ", loss_function)
        return -1

# Code provided, you don't need to modify
def threshold_predictions(predicted_values, threshold):
    '''
    Applies a threshold to the predicted values
    :param predicted_values: the predicted values
    :param threshold: the threshold
    :return: the predicted classes (thresholded predicted values)
    '''
    # threshold predictions to perform classification
    predicted_classes = np.zeros([len(predicted_values), 1])
    predicted_classes[:, 0] = predicted_values[:, 0]
    low_values_flags = predicted_values[:] < threshold
    predicted_classes[:][low_values_flags] = 0
    high_values_flags = predicted_values[:] >= threshold
    predicted_classes[:][high_values_flags] = 1

    return predicted_classes


# Code provided, you don't need to modify, but you may want to where "transform to get predictions" is
def generate_regression_line(x_start, x_stop, theta, log_on=False, degree=1):
    '''
    Generates a regression line for plotting
    :param data: the x-data of the line
    :param theta: the model
    :param log_on: True or False indicating whether to predict with a logarithmic function
    :param degree: the degree of the polynomial
    :return: a tuple containing x and y values of the line to plot
    '''
    # generate points on the line to plot
    interval = (x_stop - x_start) / 50
    line_x_values = np.arange(x_start, x_stop, interval)
    line_x = np.zeros([line_x_values.size, 1])
    line_x[:, 0] = np.arange(x_start, x_stop, interval)

    # transform to get predictions
    X_new = poly_basis(line_x, degree)
    X_new, _, _ = normalize(X_new)
    X_new = add_bias(X_new)
    # get y-values (predictions)
    line_y = predict(theta, X_new, log_on=log_on)

    return line_x, line_y

def experiment_1():
    '''
    Experiment 1 - linear regression
    Implement all the necessary functions and answer the following questions
    1) At first, there are different solutions for the analytical and gradient descent solution methods. Why? Report
       the graph and how you made it happen
       =    There are different solutions because no adjustments had been made to the learning rate. I did not need to do
            anything other than standard gradient descent to make this happen.

    2) Make the gradient descent and analytical methods produce the same solution. How did you do this? Report the
       graph and how you made it happen
       =    The learning rate needed to be adjusted in order for gradient descent to accurately model the data. I raised
            the learning rate from 0.01 to 0.1 to achieve a good fit.

    3) For Z-scaling, why is there an option to pass in a mean and standard deviation rather than calculate it from the
       data every time?
       = The shape of a normal curve is determined by its mean and standard deviation. It makes sense to calculate this from
        the data; however, you should be able to define your own so that you can make the normal curve into a desired shape.
    '''

    # degree of polynomial to fit
    max_epoch = 50
    learning_rate = 0.01 #0.01
    loss_function = "MSE"

    # create data
    data, labels = data_generator.generate_experiment_1_data()

    # set up X for regression
    X, mean, stdev = normalize(data)
    X = add_bias(X)

    # calculate theta and loss analytically
    theta_analytical = train_regression_analytical(X, labels)
    loss = calculate_loss(X, labels, theta_analytical, loss_function)

    # calculate theta and loss with gradient descent
    theta_gradient_descent = train_gradient_descent(X, labels, max_epoch, learning_rate)
    loss = calculate_loss(X, labels, theta_gradient_descent, loss_function)
    print("Model Loss analytical = ", loss)
    print("Model Loss gradient descent = ", loss)
    # print(X.shape)

    # create data to plot
    start = min(data[:, 0])
    stop = max(data[:, 0])
    # print(theta_analytical.shape)
    # print(theta_gradient_descent.shape)
    line_x_a, line_y_values_a = generate_regression_line(start, stop, theta_analytical)
    line_x_g, line_y_values_g = generate_regression_line(start, stop, theta_gradient_descent)

    # plot the samples
    fig = plt.figure()
    plt.plot(data, labels, 'b.', label='data points')
    plt.plot(line_x_a[:, 0], line_y_values_a[:, 0], 'r-', label='analytical solution')
    plt.plot(line_x_g[:, 0], line_y_values_g[:, 0], 'g-', label='gradient descent solution')
    plt.xlabel('Feature Values')
    plt.ylabel('Predicted Values')
    plt.title(' Linear Regression ')
    plt.tight_layout()  # Needed to avoid x/y label cutoff
    plt.grid(True, lw=0.5)  # By default, xkcd has lw=0.0
    plt.legend()
    fig.savefig("experiment_1")


def experiment_2():
    '''
    Experiment 2 - logistic regression (classification)
    Implement all the necessary functions and answer the following questions
    1) How does loss change as the algorithm trains? How does loss relate to the fit of the data? Show this with
       pictures and report the loss
        = The loss decreases as the algorithm trains.  The less loss there is, the better fit to the data.

    2) What would happen if you decreased the learning rate? Show these effects with pictures and report the loss
        = Decreasing the learning rate makes the fit to the data worse. Picture is attached.

    3) Why is the loss type Binary Cross Entropy (BCE) rather than Mean Squared Error (MSE)?
        = Binary cross entropy is better for classification, while mean squared error is better
            for regression.

    4) Look at the "threshold_predictions" function what is it doing? Why is it needed for classification but not
       regression?
        = It creates a threshold or boundary for the predicted values. If any of the predicted values are greater than the threshold,
         then they are changed to be either 0 or 1. This is needed in classification but not regression because in classification
         the data can be separated or grouped further away than in typical regression problems.
    '''
    learning_rate = 0.00000001 #0.0000001
    threshold = 0.5
    loss_function = "BCE"

    # create data
    data, labels = data_generator.generate_experiment_2_data()

    # set up X for regression
    X, mean, theta = normalize(data)
    X = add_bias(X)

    # you cannot calculate an analytical solution with logistic regression
    # calculate theta and loss with gradient descent
    theta_0 = train_gradient_descent(X, labels, 0, learning_rate, log_on=True)
    loss = calculate_loss(X, labels, theta_0, loss_function, log_on=True)
    print("Loss after 0 epochs = ", loss)

    theta_1 = train_gradient_descent(X, labels, 5, learning_rate, log_on=True)
    loss = calculate_loss(X, labels, theta_1, loss_function, log_on=True)
    print("Loss after 5 epoch = ", loss)

    theta_2 = train_gradient_descent(X, labels, 50, learning_rate, log_on=True)
    loss = calculate_loss(X, labels, theta_2, loss_function, log_on=True)
    print("Loss after 50 epochs = ", loss)

    theta_3 = train_gradient_descent(X, labels, 100, learning_rate, log_on=True)
    loss = calculate_loss(X, labels, theta_3, loss_function, log_on=True)
    print("Loss after 100 epochs = ", loss)

    theta_4 = train_gradient_descent(X, labels, 1000, learning_rate, log_on=True)
    loss = calculate_loss(X, labels, theta_4, loss_function, log_on=True)
    print("Loss after 1000 epochs = ", loss)

    theta_5 = train_gradient_descent(X, labels, 5000, learning_rate, log_on=True)
    loss = calculate_loss(X, labels, theta_5, loss_function, log_on=True)
    print("Loss after 5000 epochs = ", loss)

    # create data to plot
    start = min(data[:, 0])
    stop = max(data[:, 0])
    line_x_0, line_y_values_0 = generate_regression_line(start, stop, theta_0, log_on=True)
    line_x_1, line_y_values_1 = generate_regression_line(start, stop, theta_1, log_on=True)
    line_x_5, line_y_values_5 = generate_regression_line(start, stop, theta_2, log_on=True)
    line_x_25, line_y_values_25 = generate_regression_line(start, stop, theta_2, log_on=True)
    line_x_100, line_y_values_100 = generate_regression_line(start, stop, theta_4, log_on=True)
    line_x_1000, line_y_values_1000 = generate_regression_line(start, stop, theta_5, log_on=True)
    line_y_classes = threshold_predictions(line_y_values_1000, threshold)

    # plot the samples
    fig = plt.figure()
    plt.plot(data, labels, 'b.', label='data points')
    plt.plot(line_x_0[:, 0], line_y_values_0[:, 0], c='red', label='regression line at epoch 0')
    plt.plot(line_x_1[:, 0], line_y_values_1[:, 0], c='orange', label='regression line at epoch 5')
    plt.plot(line_x_5[:, 0], line_y_values_5[:, 0], c='yellow', label='regression line at epoch 50')
    plt.plot(line_x_25[:, 0], line_y_values_25[:, 0], c='green', label='regression line at epoch 100')
    plt.plot(line_x_100[:, 0], line_y_values_100[:, 0], c='blue', label='regression line at epoch 1000')
    plt.plot(line_x_1000[:, 0], line_y_values_1000[:, 0], c='purple', label='regression line at epoch 5000')
    plt.plot(line_x_0[:, 0], line_y_classes[:, 0], c='black', label='predicted values at epoch 5000')
    plt.xlabel('Feature Values')
    plt.ylabel('Predicted Values')
    plt.title(' Logistic Regression ')
    plt.tight_layout()  # Needed to avoid x/y label cutoff
    plt.grid(True, lw=0.5)  # By default, xkcd has lw=0.0
    plt.legend()
    fig.savefig("experiment_2")


def experiment_3():
    '''
     Experiment 3 - polynomial regression
     Implement all the necessary functions and answer the following questions
     1) Notice the order in which we transform the input. First we apply the basis function, then we normalize, then we
        add a bias. What would happen if we did things in a different order?
        = You cannot add the bias or normalize first because they reference X. If you change the order of
            normalize and add_bias, then it encounters an invalid value error during normalization b/c it divides by the
            standard deviation, which is zero.

     2) What happens if you don't normalize the data?
     Note: to play with the order the data is transformed or to turn off normalization, you will have to mirror your
            changes in the "generate_regression_line" function in order to generate appropriate plots
        = If the data is not normalized, the algorithm still does a good job of fitting the data; however, there is a higher
            loss than if the data was normalized. So, normalizing the data results in a better fit.

     3) Submit your plot, report the hyper-parameters you used, and the loss you acheived
        = ('Loss = ', 16789.490091087348)
            degree = 2
            loss_function = MSE
    '''

    # degree of polynomial to fit
    degree = 2  # degree of polynomial
    loss_function = 'MSE'

    # create data
    data, labels = data_generator.generate_experiment_3_data()

    # set up X for regression
    X = poly_basis(data, degree)
    X, mean, stdev = normalize(X)
    X = add_bias(X)

    # calculate theta and loss analytically
    theta = train_regression_analytical(X, labels)
    loss = calculate_loss(X, labels, theta, loss_function)
    print ("Loss = ", loss)

    # create data to plot
    start = min(data[:, 0])
    stop = max(data[:, 0])
    line_x, line_y_values = generate_regression_line(start, stop, theta, degree=degree)

    # plot the samples
    fig = plt.figure()
    plt.plot(data, labels, 'b.', label='data points')
    plt.plot(line_x[:, 0], line_y_values[:, 0], 'r-', label='analytical solution')
    plt.xlabel('Feature Values')
    plt.ylabel('Predicted Values')
    plt.title(' Linear Regression ')
    plt.tight_layout()  # Needed to avoid x/y label cutoff
    plt.grid(True, lw=0.5)  # By default, xkcd has lw=0.0
    plt.legend()
    fig.savefig("experiment_3")


def experiment_4():
    '''
     Experiment 4 - polynomial classification
     Implement all the necessary functions and answer the following questions. Your code should work with arbitrary size
     and dimension data.
     1) Get 100% classification accuracy with and without using a logistic function.
        = Plots submitted to scholar

     2) Submit both plots and describe the hyper-parameters that you used
        = Plots submitted to scholar. I first changed the degree polynomial to 2, then I tuned the max epoch so that it would train longer.
        I adjusted the learning rate to 0.1 and slightly changed the threshold value, so that it would not miss any points.
        Turned log on and off for question.

     3) Did you get an error when calculating Binary Cross Entropy (BCE) with log_on=False? Why? Did you make a change
        to get your program to work?
        = Yes, you get a 'RuntimeWarning: invalid value' error with the log function. If you change the loss function to
            MSE instead of BCE, then this fixes the error. This is because you cannot take the log negative values, and in
            this case, using BCE results in negatives in the predicted values.
    '''

    # degree of polynomial to fit
    degree = 2  # degree of polynomial
    max_epoch = 1000
    learning_rate = 0.1
    threshold = 0.64
    log_on = True  # turn this off to turn off logistic functions
    loss_function = 'BCE'

    # create data
    data, labels = data_generator.generate_experiment_4_data()

    # set up X for regression
    X = poly_basis(data, degree)
    X, mean, stdev = normalize(X)
    X = add_bias(X)

    # calculate theta and loss analytically
    theta = train_gradient_descent(X, labels, max_epoch, learning_rate, log_on=log_on)
    loss = calculate_loss(X, labels, theta, loss_function, log_on=log_on)
    print ("Loss = ", loss)

    # create data to plot
    start = min(data[:, 0])
    stop = max(data[:, 0])
    line_x, line_y_values = generate_regression_line(start, stop, theta, degree=degree, log_on=log_on)
    line_y_classes = threshold_predictions(line_y_values, threshold)

    # plot the samples
    fig = plt.figure()
    plt.plot(data, labels, 'b.', label='data points')
    plt.plot(line_x[:, 0], line_y_values[:, 0], 'r-', label='regression line')
    plt.plot(line_x[:, 0], line_y_classes[:, 0], 'g-', label='predicted values')
    plt.xlabel('X-axis')
    plt.ylabel('Y-Axis')
    plt.title(' Polynomial Classification ')
    plt.tight_layout()  # Needed to avoid x/y label cutoff
    plt.grid(True, lw=0.5)  # By default, xkcd has lw=0.0
    plt.legend()
    fig.savefig("experiment_4")


def experiment_5():
    '''
    Experiment 5 - 3D polynomial classification
    Implement all the necessary functions and answer the following questions
    1) How high accuracy can you get? Play with the hyper-parameters (I don't know if you can get 100%,
       but you can pretty easily get over 90%, with some more work, over 95%).
        = Got it up to 92.2%, see attached picture
    2) Submit your plot with the best accuracy, the accuracy, and the hyper-parameters you used to get the results
        = Plot attached
    3) Are you getting any runtime errors related to your logistic functions? Why, would you? what assumptions are
       broken? How can you solve this?
        = RuntimeWarning: divide by zero encountered in log. I solved this by messing with the hyperparemeters and making
        sure that no zeroes were passed to the log function.
    '''
    #Put learning rate really low

    # # Hyper parameters
    degree = 2  # degree of polynomial
    max_epoch = 1000
    learning_rate = 0.00000000000001 #0.001
    log_on = True
    penalty = 0
    regularization_type = 'L1'
    loss_function = 'BCE'

    # create data
    data, labels = data_generator.generate_experiment_5_data()

    # set up X for regression
    X = poly_basis(data, degree)
    X, _, _ = normalize(X)
    X = add_bias(X)

    # perform regression
    theta = train_gradient_descent(X, labels, max_epoch, learning_rate, log_on=log_on, penalty=penalty,
                                   regularization=regularization_type)
    loss = calculate_loss(X, labels, theta, loss_function, log_on=log_on, penalty=penalty,
                          regularization_type=regularization_type)
    print("Model Loss analytical = ", loss)

    # perform prediction (to generate surface)
    n, d = X.shape
    surf = predict(theta, X, log_on=log_on)
    surf_plot = np.zeros([n, 3])
    surf_plot[:, 0:2] = data
    surf_plot[:, 2] = surf[:, 0]

    # threshold prediction
    pred_plot = np.zeros([n, 3])
    pred_plot[:, 0:2] = data
    pred_plot[:, 2] = surf[:, 0]

    # pred_plot=surf_plot
    low_values_flags = pred_plot[:, 2] < 0.5
    pred_plot[:, 2][low_values_flags] = 0
    high_values_flags = pred_plot[:, 2] >= 0.5
    pred_plot[:, 2][high_values_flags] = 1

    # calculate the accuracy
    equals_matrix = pred_plot[:, 2] == labels[:, 0]
    print("Accuracy = ", float(np.sum(equals_matrix)) / float(labels.size))

    # Plot the data points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], labels[:, 0], c='r', marker='.', label='data points')
    ax.scatter(surf_plot[:, 0], surf_plot[:, 1], surf_plot[:, 2], c='g', marker='.', label='regression values')
    ax.scatter(pred_plot[:, 0], pred_plot[:, 1], pred_plot[:, 2], c='b', marker='.', label='classification values')
    plt.legend()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    fig.savefig("3D Polynomial Regression")


def experiment_6():
    '''
    Experiment 6 - Regularization
    Implement all the necessary functions and answer the following questions
    1) Why aren't we creating a plot for an analytical solution and L1 regularization?
        = We do not calculate l1 regularization with an analytical solution. It is nonlinear so l1 regularization would not work.

    2) Set the regularization penalty to 0.1 - what happens? Report your graph and loss. Explain
        = The models fit the data well. The L2 regularization has a higher loss than the other models. This is because the penalty
        affects the L2 regularization the most. The L1 regularization also had a higher loss due to the penalty.

    3) What happens to the regularized solutions when you set penalty high (1.0) and what happens when you set it low
       (0.00001)? Report your graphs and loss. Explain
       = When you set the penalty to high, the L2 regularization becomes a much worse fit and the other lines deviate slightly.
        When you set the penalty to low, there is little deviation and all of the graphs line up well and fit the model.
        The loss for L2 was the highest, followed by L1 for each case. In the case of penalty being 1, the L2 loss was significantly
        higher than the others. This is because the penalty value affects L2 more, since it is penalty*theta(T)*theta instead of just
        theta * abs(theta).
    '''

    # Hyper parameters
    degree = 4  # degree of polynomial
    max_epoch = 5000
    learning_rate = 0.01
    penalty = 0.00001
    loss_function = 'MSE'

    # create data
    data, labels = data_generator.generate_experiment_6_data()

    # set up X for regression
    X = poly_basis(data, degree)
    X, mean, stdev = normalize(X)
    X = add_bias(X)

    # calculate theta and loss analytically
    theta_analytical_no_reg = train_regression_analytical(X, labels)
    loss = calculate_loss(X, labels, theta_analytical_no_reg, loss_function)
    print("Model Loss analytical no regularization = ", loss)

    # calculate theta and loss analytically (with L2 regularization)
    theta_analytical_reg = train_regression_analytical(X, labels, regularization='l2', penalty=penalty)
    loss = calculate_loss(X, labels, theta_analytical_reg, loss_function)
    print("Model Loss analytical with L2 regularization = ", loss)

    # calculate theta and loss with gradient descent (with L2 regularization)
    theta_gradient_reg_l2 = train_gradient_descent(X, labels, max_epoch, learning_rate, regularization='l2',
                                                   penalty=penalty)
    loss = calculate_loss(X, labels, theta_gradient_reg_l2, loss_function)
    print("Model Loss gradient descent with L2 regularization = ", loss)

    # calculate theta and loss with gradient descent (with L1 regularization)
    theta_gradient_reg_l1 = train_gradient_descent(X, labels, max_epoch, learning_rate, regularization='l1',
                                                   penalty=penalty)
    loss = calculate_loss(X, labels, theta_gradient_reg_l1, loss_function)
    print("Model Loss gradient descent with L1 regularization = ", loss)

    # create data to plot
    start = min(data[:, 0])
    stop = max(data[:, 0])
    line_x_an, line_y_values_an = generate_regression_line(start, stop, theta_analytical_no_reg, degree=degree)
    line_x_ar, line_y_values_ar = generate_regression_line(start, stop, theta_analytical_reg, degree=degree)
    line_x_gr2, line_y_values_gr2 = generate_regression_line(start, stop, theta_gradient_reg_l2, degree=degree)
    line_x_gr1, line_y_values_gr1 = generate_regression_line(start, stop, theta_gradient_reg_l1, degree=degree)

    # plot the samples
    fig = plt.figure()
    plt.plot(data, labels, 'b.', label='data points')
    plt.plot(line_x_an[:, 0], line_y_values_an[:, 0], 'r-', label='analytical solution no regularization')
    plt.plot(line_x_ar[:, 0], line_y_values_ar[:, 0], c='orange', label='analytical solution with L2 regularization')
    plt.plot(line_x_gr2[:, 0], line_y_values_gr2[:, 0], c='green',
             label='gradient descent solution with L2 regularization')
    plt.plot(line_x_gr1[:, 0], line_y_values_gr1[:, 0], c='blue',
             label='gradient descent solution with L1 regularization')
    plt.xlabel('Feature Values')
    plt.ylabel('Predicted Values')
    plt.title(' Regularization ')
    plt.tight_layout()  # Needed to avoid x/y label cutoff
    plt.grid(True, lw=0.5)  # By default, xkcd has lw=0.0
    plt.legend()
    fig.savefig("experiment_6")


#############################################################
#                 Run the Script
#############################################################
if __name__ == '__main__':
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
    experiment_5()
    experiment_6()
    plt.show()
