import numpy as np

from regression import rbf, rbf_pca
from regression import GPy_Regressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Load data
data = np.load('reduced_trajectories.pkl',allow_pickle=True,encoding='bytes')
wps_data = data.get('waypoints')
timings = data.get('timings')
targets_data= data.get('targets')

nq = wps_data[0].shape[1]
nx = nq * 2
num_trajs = len(wps_data)
max_time = 5
T_horizon = 100

dt = 0.01

'''
# Dimensionality reduction
trajs_q_full = []
for i in range(num_trajs_full):
	traj_q_full = []
	for j in range(T_horizon):
		traj_q_full.append(trajs_full[i][j][:29])
	trajs_q_full.append(np.array(traj_q_full))

trajs_q_full = np.array(trajs_q_full)'''

gpr_inputs = []
gpr_outputs = []
for t in range(num_trajs):
	print(f"Compute traj {t}")
	tnow = 0
	xs = []
	qprec = wps_data[t][0]
	for i in range(wps_data[t].shape[0]):
		while tnow < max_time:
			qnow = (timings[t][i+1] - tnow) / (timings[t][i+1] - timings[t][i]) * wps_data[t][i] +\
				(tnow - timings[t][i]) / (timings[t][i+1] - timings[t][i]) * wps_data[t][i+1]
			xs.append(np.concatenate((qnow,(qnow - qprec) * 100)))
			qprec = qnow
			tnow += dt
	recede = 0
	if len(xs) <= T_horizon:
		xs = xs + [xs[-1] for _ in range(T_horizon - len(xs))]
		gpr_inputs.append(np.concatenate((np.array(targets_data[t]),wps_data[t][0])))
		gpr_outputs.append(np.array(xs))
		continue
	while ((recede + T_horizon) < len(xs)):
		piece_traj = xs[recede:recede + T_horizon]
		gpr_inputs.append(np.concatenate((np.array(targets_data[t]),xs[recede][:nq])))
		gpr_outputs.append(np.array(piece_traj))
		recede += 1

# Train-test split
gpr_inputs = np.array(gpr_inputs)
gpr_outputs = np.array(gpr_outputs)

rbf = rbf( K = 30, offset = 10, width = 1.75, T = T_horizon + 1, reg_factor=1e-4)
rbf.create_RBF()
pca = PCA(n_components=30)
rbf_pca_fun = rbf_pca(rbf, pca)
trajs_pca = rbf_pca_fun.transform(gpr_outputs)

x_train,x_test, y_train, y_test = train_test_split(gpr_inputs, trajs_pca.reshape(num_trajs, -1), test_size=0.3) 

# Train state GPR
gpr = GPy_Regressor(dim_input = x_train.shape[1],transform=rbf_pca_fun) #rbf_pca_fun
gpr.fit(x_train, y_train, num_restarts=3)
gpr.save_to_file('data/gpr_reduced')
print(gpr.gp.flattened_parameters)

# Validate state GPR
gpr_errs = []
for i,x in enumerate(x_test):
    y = y_test[i]
    y_pred, _ = gpr.predict(x[None,:], False)
    gpr_errs += [np.linalg.norm(y-y_pred)]

print(f"Mean GPR error {np.mean(gpr_errs)}, STD GPR error {np.std(gpr_errs)}")
