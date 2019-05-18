
from sklearn.ensemble import RandomForestClassifier
import numpy
from joblib import Parallel, delayed
from numba import jit




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




def is_good_vec(tree, X):
    """
    """
    is_good = (tree.predict_proba(X)[:, 0] > 0.5)
    return  is_good


def is_good_matrix_get(forest, X):
    is_good_matrix = Parallel(n_jobs=-1, verbose=0)(delayed(is_good_vec)
                                                    (tree, X) for tree in forest.estimators_)
    is_good_matrix = numpy.vstack(is_good_matrix)
    is_good_matrix = is_good_matrix.T

    return is_good_matrix

@jit
def get_anomaly_score_slow(leafs, is_good, fe):

    start = fe[0]
    end = fe[1]

    obs_num = leafs.shape[0]
    tree_num = leafs.shape[1]
    anomaly_score = numpy.ones(end-start,dtype = numpy.dtype('f4'))
    dists = numpy.ones(obs_num,dtype = numpy.dtype('f4'))


    for i in range(start,end):
        for j in range(obs_num):
            same_leaf = 0
            good_trees = 0
            for k in range(tree_num):
                if (is_good[i,k]  == 1 ) and (is_good[j,k] == 1):
                    good_trees = good_trees + 1
                    if (leafs[i,k] == leafs[j,k]):
                        same_leaf = same_leaf + 1
            if good_trees == 0:
                dis = 1
            else:
                dis = 1 - float(same_leaf) / good_trees

            dists[j] = dis
        anomaly_score[i] = numpy.sum(dists)

    return anomaly_score

@jit
def build_distance_matrix_slow(leafs, is_good, fe):

    start = fe[0]
    end = fe[1]

    obs_num = leafs.shape[0]
    tree_num = leafs.shape[1]
    dis_mat = numpy.ones((end - start,obs_num),dtype = numpy.dtype('f4'))

    for i in range(start,end):
        jstart = i
        for j in range(jstart, obs_num):
            same_leaf = 0
            good_trees = 0
            for k in range(tree_num):
                if (is_good[i,k]  == 1 ) and (is_good[j,k] == 1):
                    good_trees = good_trees + 1
                    if (leafs[i,k] == leafs[j,k]):
                        same_leaf = same_leaf + 1
            if good_trees == 0:
                dis = 1
            else:
                dis = 1 - float(same_leaf) / good_trees

            dis_mat[i - start][j] = dis

    return dis_mat

@jit
def distance_mat_fill(dis_mat):


    for i in range(len(dis_mat)):
        jend = i
        for j in range(0,jend):

            dis_mat[i][j] = dis_mat[j][i]

    return dis_mat

class urf(object):
    def __init__(self, n_trees = 100, synthetic_data_type = None, max_features='auto', max_depth=None):

        self.n_trees = n_trees
        self.synthetic_data_type = synthetic_data_type
        self.max_features = max_features
        self.max_depth = max_depth

        return


    def get_random_forest(self):
        """
        Runs random forest on X
        """
        X_total, Y_total = create_synthetic_data(self.X, self.synthetic_data_type)


        rf = RandomForestClassifier(n_jobs=-1, n_estimators=self.n_trees, max_features = self.max_features,  max_depth = self.max_depth)
        rf.fit(X_total, Y_total)

        return rf

    def get_leafs(self):

        rf = self.get_random_forest()
        rf_leafs = rf.apply(self.X)

        is_good = is_good_matrix_get(rf, self.X)
        return rf_leafs, is_good

    def get_Xs(self,X ):
        try:
            objnum = X.shape[0]
            Xs = X
        except:

            Xs = X.copy()
            X = numpy.hstack(Xs)
            objnum = X.shape[0]

        csize = 10
        start = numpy.arange(1 + int(objnum / csize)) * csize
        end = start + csize
        fe = numpy.vstack([start, end]).T
        fe[-1][1] = objnum

        self.fe = fe
        self.Xs = Xs
        self.X = X

        return


    def get_distance(self,X):
        self.get_Xs(X)
        rf_leafs, is_good = self.get_leafs()

        distance_matrix = Parallel(n_jobs=-1)(delayed(build_distance_matrix_slow)
                                              (rf_leafs, is_good, se)          for se in self.fe)
        distance_matrix = numpy.vstack(distance_matrix)

        distance_matrix = distance_mat_fill(distance_matrix)

        return distance_matrix


    def get_anomaly_score(self,X):
        self.get_Xs(X)
        rf_leafs, is_good = self.get_leafs()

        anomaly_score = Parallel(n_jobs=-1)(delayed(get_anomaly_score_slow)
                                              (rf_leafs, is_good, se)          for se in self.fe)
        anomaly_score = numpy.concatenate(anomaly_score)


        return anomaly_score
