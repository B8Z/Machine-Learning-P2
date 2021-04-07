import numpy as np

def generate_experiment_1_data():
    # generate a distribution
    start = -10
    stop = 10
    interval = 0.5
    variance = 1.5

    a = 0.6
    b = 3
    x_values = np.arange(start, stop, interval)
    y_values = []
    for x in x_values:
        y_values.append(a*x+b + variance*np.random.randn())

    #format correctly
    n = len(y_values)
    data = np.zeros([n, 1])
    labels = np.zeros([n, 1])
    data[:, 0] = x_values
    labels[:, 0] = y_values

    return data, labels



def generate_experiment_2_data():
    # generate a distribution
    start = -10
    stop = 10
    interval = 0.5

    x_values = np.arange(start, stop, interval)
    y_values = []
    for x in x_values:
        if (x > 0):
            y_values.append(0)
        else:
            y_values.append(1)

    # format correctly
    n = len(y_values)
    data = np.zeros([n, 1])
    labels = np.zeros([n, 1])
    data[:, 0] = x_values
    labels[:, 0] = y_values

    return data, labels

def generate_experiment_3_data():
    # generate a distribution
    start = -10
    stop = 10
    interval = 0.5
    variance = 100

    c = 1
    a = 1
    b = 1
    d = 0
    x_values = np.arange(start, stop, interval)
    y_values = []
    for x in x_values:
        y_values.append(a*x*x*x + b*x*x + c*x + d + variance*np.random.randn())

    #format correctly
    n = len(y_values)
    data = np.zeros([n, 1])
    labels = np.zeros([n, 1])
    data[:, 0] = x_values
    labels[:, 0] = y_values

    return data, labels

def generate_experiment_4_data():
    # generate a distribution
    start = -10
    stop = 10
    interval = 0.5

    x_values = np.arange(start, stop, interval)
    y_values = []
    for x in x_values:
        if (x < -3):
            y_values.append(0)
        elif (x >= -3 and x <= 3):
            y_values.append(1)
        else:
            y_values.append(0)

    # format correctly
    n = len(y_values)
    data = np.zeros([n, 1])
    labels = np.zeros([n, 1])
    data[:, 0] = x_values
    labels[:, 0] = y_values

    return data, labels



def generate_experiment_5_data():
    # generate a distribution
    start = -10
    stop = 10
    interval = 0.5
    x_variance = 5000
    y_variance = 10000
    n = int((stop - start) / (interval))
    n = n * n  # since its two-dimensional we create an nxn grid

    # True function weights
    a1 = 0.1
    b1 = 0.4
    c1 = 0.01
    d1 = 1

    a2 = 0.8
    b2 = 0.03
    c2 = 0.3
    d2 = 0
    x_points = np.arange(start, stop, interval)
    y_points = np.arange(start, stop, interval)
    z_values = []
    x_values = []
    y_values = []
    for x in x_points:
        for y in y_points:
            # set the class
            z = 1
            if x < 3 and y > -2:
                z = 0

            if (x < 1 and x > -1 and y > 7 and y < 10):
                z = 1

            x_values.append(x)
            y_values.append(y)
            z_values.append(z)

    # combine the dimensions into a matrix
    all_data = np.zeros([n, 3])
    all_data[:, 0] = x_values
    all_data[:, 1] = y_values
    all_data[:, 2] = z_values

    # generate the labels
    labels = np.zeros([n, 1])
    labels[:, 0] = z_values

    # split into dat and labels
    data = all_data[:, :-1]
    labels = np.zeros([n, 1])
    labels[:, 0] = all_data[:, -1]

    return data, labels


def generate_experiment_6_data():
    # generate a distribution
    start = -10
    stop = 10
    interval = 0.5
    variance = 300

    c = 5
    a = -8
    b = 23
    d = 52
    x_values = np.arange(start, stop, interval)
    y_values = []
    for x in x_values:
        y_values.append(a*x*x*x + b*x*x + c*x + d + variance*np.random.randn())
    y_values[-10]= -5500
    y_values[-9] = -5000

    y_values[5] = 12000
    y_values[6] = 13000

    #format correctly
    n = len(y_values)
    data = np.zeros([n, 1])
    labels = np.zeros([n, 1])
    data[:, 0] = x_values
    labels[:, 0] = y_values

    return data, labels
