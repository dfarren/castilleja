import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state
from sklearn.linear_model import SGDRegressor, LinearRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
import pdb


def generate_linear_data(slope, intercept, noise_std, max_x=100):
    x = np.arange(max_x)
    x = np.reshape(x, (max_x,1))
    y = np.array([i*slope+intercept+np.random.normal(0, noise_std) for i in x])
    return x, np.ravel(y)


def generate_quadratic_data(a, b, c, noise_std, min_x=0, max_x=100):
    x = np.reshape(np.arange(max_x), (max_x,1))
    x = np.reshape(x, (max_x,1))
    y = np.array([a*i**2+b*i+c+np.random.normal(0, noise_std) for i in x])
    return x, np.ravel(y)


def generate_unscaled_data(coefs, intercept, noise_std, min_x=-100, max_x=100, scale=100):
    x1 = np.array([[e, 0] for e in np.arange(min_x, max_x)])
    x2 = np.array([[0, e*scale] for e in np.arange(min_x, max_x)])
    x = np.vstack((x1, x2))
    y = np.array([x1**2*coefs[0]+x2**2*coefs[1]+intercept+np.random.normal(0, noise_std) for x1, x2 in x])
    pdb.set_trace()
    return x, np.ravel(y)


def plot(x, y, label=[], predictions=[], x_boundary=[], y_boundary=[], x_label='x', y_label='y', line=False, line_weight=4):

    x = np.array(x)
    y = np.array(y)
    # 2D plot
    if len(x.shape)==1 or x.shape[1]==1:
        fig, ax = plt.subplots()

        if line:
            ax.plot(x,y, lw=line_weight)
        elif label != []:
            colors = []
            for label in label:
                if label==1:
                    colors.append('blue')
                else:
                    colors.append('red')
            ax.scatter(x, y, c=colors, s=50)
        else:
            ax.scatter(x, y)

        if predictions != []:
            predictions = np.array(predictions)
            ax.plot(x, predictions, 'r-', lw=line_weight)

        if x_boundary != []:
            x_boundary = np.array(x_boundary)
            y_boundary = np.array(y_boundary)
            ax.plot(x_boundary, y_boundary, 'r-', lw=line_weight)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.show()

    # 3D plot
    elif x.shape[1]==2:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = x[:, 0]
        Y = x[:, 1]
        X, Y = np.meshgrid(X, Y)
        Z = y
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        #ax.set_zlim(-1.01, 1.01)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    return


def print_residuals(y, predictions):
    predictions = predictions.ravel()
    residuals = np.sum(np.abs(y-predictions))
    print("Summation of residuals is {:,.2f}".format(residuals))


def loss(y, predictions, delta, function='least_squares'):
    '''
    :param x: numpy array with data
    :param delta: the slope
    :return: huber function of x
    '''
    x = (y-predictions).astype(float)

    if function=='least_squares':
        pass
    elif function=='huber':
        idx_squared = np.nonzero(np.abs(x)<=delta)
        idx_linear = np.nonzero(np.abs(x)>delta)
        x[idx_squared] = (x[idx_squared]**2)/2
        x[idx_linear] = delta*(np.abs(x[idx_linear])-delta/2)
    return x
    

def huber(x, delta):
    '''
    :param x: numpy array with data
    :param delta: the threshold
    :return: huber function of x
    '''
    x = x.astype(float)
    idx_squared = np.nonzero(np.abs(x)<=delta)
    idx_linear = np.nonzero(np.abs(x)>=delta)
    x[idx_squared] = (x[idx_squared]**2)/2
    x[idx_linear] = delta*(np.abs(x[idx_linear])-delta/2)

    return x

# def plot_levels(X,y):
#     fig, ax = plt.subplots()
#     ax.scatter(x, y)
#     if isinstance(predictions, np.ndarray):
#         ax.plot(x, predictions, 'r-', lw=4)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     plt.show()
#     return

def faces_data():

    data = fetch_olivetti_faces()
    targets = data.target

    data = data.images.reshape((len(data.images), -1))
    train = data[targets >= 10]
    test = data[targets < 10 ]  # Test on independent people

    # Test on a subset of people
    n_faces = 5
    rng = check_random_state(np.random)
    face_ids = rng.randint(test.shape[0], size=(n_faces, ))
    test = test[face_ids, :]

    n_pixels = data.shape[1]
    X_train = train[:, :np.ceil(0.5 * n_pixels)]  # Upper half of the faces
    y_train = train[:, np.floor(0.5 * n_pixels):]  # Lower half of the faces
    X_test = test[:, :np.ceil(0.5 * n_pixels)]
    y_test = test[:, np.floor(0.5 * n_pixels):]

    return (X_train, y_train), (X_test, y_test)


def plot_faces(X_test, y_test, y_test_predict):

    image_shape = (64, 64)
    n_faces = 5
    n_cols = 2
    plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
    plt.suptitle("Face completion with regression", size=16)

    for i in range(n_faces):
        true_face = np.hstack((X_test[i], y_test[i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title="true faces")


        sub.axis("off")
        sub.imshow(true_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

        completed_face = np.hstack((X_test[i], y_test_predict[i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2)

        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2,
                              title="regression")

        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

    plt.show()

def logistic(theta, x):
    theta = np.array(theta)
    x = np.array(x)
    x = np.reshape(x, (x.shape[0], 1))
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    return (1 + np.exp(-x.dot(theta)))**-1


def generate_classification_data(dim=1, size=100, noise_level=0.2):

    if dim==1:
        x = np.arange(size)
        y = np.ones(size)
        y[:size/2] = np.zeros(size/2)

    elif dim==2:
        x1 = np.arange(size)
        x2 = np.arange(size)
        np.random.shuffle(x2)
        x = (x1, x2)

        y = np.ones(size)

        #linear classification
        y[(-x1 + size)<x2] = 0

    #add noise
    idx = np.random.choice(size, int(size*noise_level))
    y[idx] = np.random.randint(2, size=len(idx))

    return x, y


def logistic_regression_boundary(model, size=100):
    '''
    works only for two features dataset
    '''

    if isinstance(model, SGDClassifier):
        theta = np.append(model.intercept_, model.coef_)
    else:
        theta = model

    x1 = np.arange(size)
    x2 = (-theta[0]-x1*theta[1])/theta[2]

    return x1, x2


def hstack(x1, x2):
    return np.hstack((np.reshape(x1, (len(x1), 1)), np.reshape(x2, (len(x2), 1))))


if __name__=='__main__':

    #x, y = generate_unscaled_data(coefs=[1,1], intercept=1, noise_std=10, scale=10)
    #plot(x, y)

    (x1, x2), y = generate_classification_data(dim=2, noise_level=0.1)

    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression()
    x = np.hstack((np.reshape(x1, (len(x1), 1)), np.reshape(x2, (len(x2), 1))))

    m.fit(x, y)

    theta = np.append(m.intercept_, m.coef_)
    x1_, x2_ = logistic_regression_boundary(theta)

    plot(x1, x2, label = y, x_boundary=x1_, y_boundary=x2_, x_label="a", y_label="b")
