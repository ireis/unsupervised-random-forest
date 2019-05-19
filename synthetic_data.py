import numpy

def create_synthetic_data(X,synthetic_data_type):
    """
    Creates synthetic data for RR dissimilarity
    :param X:
    :param kwargs:
    :return:
    """
    nof_objects = X.shape[0]
    if synthetic_data_type is None:
        synthetic_data_type = 'default'
    if synthetic_data_type == 'default':
        synthetic_X = default_synthetic_data(X)
        X_total = numpy.concatenate([X,
                                 synthetic_X])


    elif synthetic_data_type == 'f':
        synthetic_X = f_synthetic_data(X)
        X_total = numpy.concatenate([numpy.hstack(X),
                                 synthetic_X])
    else:
        print('Bad synthetic data type')
        return -1

    Y_total = numpy.concatenate([numpy.zeros(nof_objects),
                                 numpy.ones(nof_objects)])
    return X_total, Y_total

def f_synthetic_data(X_list):
    """
    Synthetic data with same marginal distribution for each feature
    """

    X = numpy.hstack(X_list)
    synthetic_X = numpy.zeros(X.shape)

    nof_chunks = len(X_list)
    nof_objects = X.shape[0]

    chunks_inds = numpy.random.choice(numpy.arange(nof_objects), [nof_objects, nof_chunks])

    for i in range(nof_objects):
        x = [X_list[c][chunks_inds[i,c]] for c in range(nof_chunks)]

        synthetic_X[i] = numpy.hstack(x)

    return synthetic_X

def default_synthetic_data(X):
    """
    Synthetic data with same marginal distribution for each feature
    """
    synthetic_X = numpy.zeros(X.shape)

    nof_features = X.shape[1]
    nof_objects = X.shape[0]

    for f in range(nof_features):
        feature_values = X[:, f]
        synthetic_X[:, f] += numpy.random.choice(feature_values, nof_objects)
    return synthetic_X
