import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch import nn
import torch.optim as optim
import tqdm
import copy 

class nn_model(nn.Module):
    def __init__(self, il, hl, ol):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(il, hl),
          nn.ReLU(),
          nn.Linear(hl, hl),
          nn.ReLU(),
          nn.Linear(hl, hl),
          nn.ReLU(),
          nn.Linear(hl, ol))
    
    def forward(self, y):
        return self.layers(y)
    
    def normalize(self,nn_inputs,nn_outputs):
        self.inputs_mean = np.mean(nn_inputs,axis=0)
        self.inputs_std = np.std(nn_inputs,axis=0)
        
        self.outputs_mean = np.mean(nn_outputs,axis=0)
        self.outputs_std = np.std(nn_outputs,axis=0)
        
        inputs_normalized = (nn_inputs - self.inputs_mean) / self.inputs_std
        outputs_normalized = (nn_outputs - self.outputs_mean) / self.outputs_std
        
        return inputs_normalized, outputs_normalized

if __name__ == '__main__':
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
	
	print(f"Traj number = {num_trajs}")

	nn_inputs = []
	nn_outputs = []
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
			nn_inputs.append(np.concatenate((np.array(targets_data[t]),wps_data[t][0])))
			nn_outputs.append(np.array(xs))
			continue
		while ((recede + T_horizon) < len(xs)):
			piece_traj = xs[recede:recede + T_horizon]
			nn_inputs.append(np.concatenate((np.array(targets_data[t]),xs[recede][:nq])))
			nn_outputs.append(np.array(piece_traj))
			recede += 1

	# Train-test split
	nn_inputs = np.array(nn_inputs)
	nn_outputs = np.array(nn_outputs)
	X_train,X_test, y_train,  y_test = train_test_split(nn_inputs, nn_outputs, test_size=0.2) 
	X_train = torch.tensor(X_train, dtype=torch.float32)
	y_train = torch.tensor(y_train, dtype=torch.float32)
	X_test = torch.tensor(X_test, dtype=torch.float32)
	y_test = torch.tensor(y_test, dtype=torch.float32)
	
	 # training parameters
	n_epochs = 1000   # number of epochs to run
	batch_size = 16  # size of each batch
	batch_start = torch.arange(0, len(X_train), batch_size)
	 
	# Define the model
	model = nn_model(nq + 3,64, nx * T_horizon)

	# Define loss function, and optimizer
	criterion = nn.MSELoss()
	#optimizer = optim.Adam(model.parameters(), lr=0.1)
	optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

	# Hold the best model
	best_mse = np.inf   # init to infinity
	best_weights = None
	history = []
	i = 0
	# training loop           
	for epoch in range(n_epochs):
		model.train()
		with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
			bar.set_description(f"Epoch {epoch}")
			for start in bar:
				# take a batch
				X_batch = X_train[start:start+batch_size]
				y_batch = y_train[start:start+batch_size]
				
				# forward pass
				y_pred = model(X_batch)
				y_pred = y_pred.reshape(X_batch.shape[0],T_horizon,nx)
				loss = criterion(y_pred, y_batch)
				# backward pass
				optimizer.zero_grad()
				loss.backward()
				# update weights
				optimizer.step()
				# print progress
				bar.set_postfix(mse=float(loss))
		# evaluate accuracy at end of each epoch
		model.eval()
		y_pred = model(X_test)
		y_pred = y_pred.reshape(X_test.shape[0],T_horizon,nx)
		mse = criterion(y_pred, y_test)
		mse = float(mse)
		history.append(mse)
		if mse < best_mse:
			best_mse = mse
			best_weights = copy.deepcopy(model.state_dict())
	 
	# restore model and return best accuracy
	model.load_state_dict(best_weights)
	torch.save(model.state_dict(),'nn_model/reduced_nn_obstacle2')


	print("MSE: %.5f" % best_mse)
	print("RMSE: %.5f" % np.sqrt(best_mse))
	plt.yscale("log")
	plt.plot(history)
	plt.show()
	
