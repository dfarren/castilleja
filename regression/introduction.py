import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
import pdb


def generate_linear_data(slope, intercept, noise_std, max_x=100):
    x = np.arange(max_x)
    y = np.array([i*slope+intercept+np.random.normal(0, noise_std) for i in x])
    x = np.reshape(x, (max_x,1))
    return x, y


def generate_quadratic_data(a, b, c, noise_std, min_x=0, max_x=100):
    x = np.reshape(np.arange(max_x), (max_x,1))
    y = np.array([a*i**2+b*i+c+np.random.normal(0, noise_std) for i in x])
    return x, y


def generate_unscaled_data(coefs, intercept, noise_std, min_x=0, max_x=100, scale=1000):
    x = np.array([[e, e*scale] for e in np.arange(max_x)])
    y = np.array([x1*coefs[0]+x2*coefs[1]+intercept+np.random.normal(0, noise_std) for x1, x2 in x])
    return x, y


def plot(x,y,predictions=None):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    if isinstance(predictions, np.ndarray):
        ax.plot(x, predictions, 'r-', lw=4)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
    return

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

#x, y = generate_linear_data(slope=1, intercept=10, noise_std=10)


