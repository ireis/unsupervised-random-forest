
from sklearn.ensemble import RandomForestClassifier
import numpy
from joblib import Parallel, delayed
from numba import njit
import synthetic_data



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
        X_total, Y_total = synthetic_data.create_synthetic_data(self.X, self.synthetic_data_type)


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


    def get_anomaly_score(self,X,mean_over=2500, knn=None):
        self.get_Xs(X)
        rf_leafs, is_good = self.get_leafs()

        nof_objects = X.shape[0]

        if not knn is None:
            mean_over = nof_objects
            knn = int(knn)

        if  mean_over < nof_objects:
            distance_to_objects = numpy.random.choice(nof_objects,mean_over,replace=False)
        else:
            distance_to_objects = numpy.arange(nof_objects)

        anomaly_score = Parallel(n_jobs=-1)(delayed(get_anomaly_score_slow)
                                              (knn, distance_to_objects, rf_leafs, is_good, se)          for se in self.fe)

        anomaly_score = numpy.concatenate(anomaly_score)


        return anomaly_score




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

@njit
def get_anomaly_score_slow(knn, distance_to_objects,leafs, is_good, fe):

    start = fe[0]
    end = fe[1]

    obs_num = leafs.shape[0]
    tree_num = leafs.shape[1]
    anomaly_score = numpy.zeros(end-start)
    dists = numpy.zeros(distance_to_objects.shape)


    for i in range(start,end):
        for j_idx, j in enumerate(distance_to_objects):
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

            dists[j_idx] = dis
        if knn is None:
            anomaly_score[i-start] = numpy.sum(dists)
        else:
            anomaly_score[i-start] = numpy.sort(dists)[knn]

    return anomaly_score

@njit
def build_distance_matrix_slow(leafs, is_good, fe):

    start = fe[0]
    end = fe[1]

    obs_num = leafs.shape[0]
    tree_num = leafs.shape[1]
    dis_mat = numpy.ones((end - start,obs_num))

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

@njit
def distance_mat_fill(dis_mat):


    for i in range(len(dis_mat)):
        jend = i
        for j in range(0,jend):

            dis_mat[i][j] = dis_mat[j][i]

    return dis_mat
