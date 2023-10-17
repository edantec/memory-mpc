import numpy as np
import numpy.matlib
import pickle
import scipy.stats
class rbf():
    def __init__(self, D=39, K=60, offset=200, width=60, T=4000, reg_factor = 1e-6):
        self.D = D
        self.K = K
        self.offset = offset
        self.width = width
        self.T = T
        self.reg_factor = reg_factor

    def create_RBF(self):
        tList = np.arange(self.T)

        Mu = np.linspace(tList[0]-self.offset, tList[-1]+self.offset, self.K)
        Sigma  = np.reshape(np.matlib.repmat(self.width, 1, self.K),[1, 1, self.K])
        Sigma.shape
        Phi = np.zeros((self.T, self.K))

        for i in range(self.K):
            Phi[:,i] = scipy.stats.norm(Mu[i], Sigma[0,0,i]).pdf(tList)

        #normalize
        Phi = Phi/np.sum(Phi,axis = 1)[:,None]
        self.Phi = Phi
        return Phi

    def transform(self,trajs):
        w_trajs = []
        for traj in trajs:
            w,_,_,_ = np.linalg.lstsq(self.Phi, traj,self.reg_factor)
            w_trajs.append(w.flatten())
        return np.array(w_trajs)

    def inverse_transform(self,ws):
        trajs = []
        for w in ws:
            w = w.reshape(self.K,-1)
            traj = np.dot(self.Phi,w)
            trajs += [traj]
        return np.array(trajs)

class rbf_pca():
    def __init__(self, rbf, pca):
        self.rbf = rbf
        self.pca = pca

    def fit_transform(self, trajs):
        w_trajs = self.rbf.transform(trajs)
        trajs_pca = self.pca.fit_transform(w_trajs)
        self.w_trajs = w_trajs
        return trajs_pca

    def transform(self, traj):
        w_traj = self.rbf.transform(traj)
        traj_pca = self.pca.fit_transform(w_traj)
        self.w_traj = w_traj
        return traj_pca

    def inverse_transform(self, trajs_pca):
        w_trajs = self.pca.inverse_transform(trajs_pca)
        trajs = self.rbf.inverse_transform(w_trajs)
        self.w_trajs = w_trajs
        return trajs

class Regressor():
    def __init__(self, transform=None):
        self.transform = transform
        self.pca = None

    def save_to_file(self,filename):
        f = open(filename + '.pkl', 'wb')
        pickle.dump(self.__dict__,f)
        f.close()

    def load_from_file(self,filename):
        f = open(filename + '.pkl', 'rb')
        self.__dict__ = pickle.load(f)

#Nearest Neighbor Regressor
from sklearn.neighbors import NearestNeighbors
class NN_Regressor(Regressor):
    def __init__(self, transform=None, K = 1):
        self.transform = transform
        self.pca = None
        self.nn = NearestNeighbors(n_neighbors=K)

    def fit(self,x,y):
        self.x = x.copy()
        self.nn.fit(x)
        self.y = y.copy()

    def nearest(self,x_i):
        # dists = []
        # for x_j in self.x:
        #     dists.append(np.linalg.norm(x_i-x_j))
        # dists = np.array(dists)
        # sort_indexes = np.argpartition(dists, self.K)
        # near_indexes, near_dists = sort_indexes[:self.K], dists[sort_indexes[:self.K]]
        near_dists, near_indexes = self.nn.kneighbors(x_i)
        return near_indexes, near_dists

    def predict(self,x, is_transform = True):
        y_indexes,dists = self.nearest(x)
        y_curs = self.y[y_indexes,:].copy()
        y = np.mean(y_curs, axis=0)
        if is_transform:
            y_transform = self.transform.inverse_transform([y[None,:]])[0]
            return y_transform, 0
        else:
            return y, 0

#GPy GP Regressor
import GPy
class GPy_Regressor(Regressor):
    def __init__(self, dim_input, transform = None):
        self.transform = transform #whether the output should be transformed or not. Possible option: PCA, RBF, etc.
        self.dim_input = dim_input

    def fit(self,x,y, num_restarts = 10):
        kernel = GPy.kern.RBF(input_dim=self.dim_input, variance=0.1,lengthscale=0.3, ARD=True) + GPy.kern.White(input_dim=self.dim_input)
        self.gp = GPy.models.GPRegression(x, y, kernel)
        self.gp.optimize_restarts(num_restarts=num_restarts)

    def predict(self,x, is_transform = True):
        y,cov = self.gp.predict(x)
        if is_transform:
            y_transform = self.transform.inverse_transform([y[None,:]])[0]
            return y_transform, cov
        else:
            return y,cov

#Sparse GP Regressor
class Sparse_GPy_Regressor(Regressor):
    def __init__(self, num_z = 100, transform = None):
        self.zdim = num_z
        self.transform = transform
    def fit(self,x,y):
        Z = x[0:self.zdim]
        self.sparse_gp = GPy.models.SparseGPRegression(x, y, Z=Z)
        self.sparse_gp.optimize('bfgs')

    def predict(self,x, is_transform = True):
        y,cov = self.sparse_gp.predict(x)
        if is_transform:
            y_transform = self.transform.inverse_transform([y[None,:]])[0]
            return y_transform, cov
        else:
            return y,cov
