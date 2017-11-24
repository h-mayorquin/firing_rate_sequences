import sklearn
import numpy as np
import sklearn.linear_model as linear_model

from network import run_network_recall, train_network, run_network_recall_limit
from connectivity import designed_matrix_sequences, designed_matrix_sequences_local
from analysis import get_recall_duration_for_pattern, get_recall_duration_sequence
from analysis import time_t1, time_t2, time_t1_local, time_t2_local, time_t2_complicated

from sklearn.model_selection import train_test_split

N = 10
tau_z = 0.150
tau_z_post = 0.005
tau_w = 0.100
max_w = 30.0
min_w = -3.0

training_time = 0.100
inter_sequence_time = 1.000
dt = 0.001
epochs = 4
sequence1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
sequences = [sequence1]

pattern = 3
pattern_from = 2

saved_matrix = True
outfile = './data1.npz'
if not saved_matrix:
    n_samples = 5000
    max_w_vector = np.random.uniform(low=1, high=200, size=n_samples)
    min_w_vector = -np.random.uniform(low=1, high=100, size=n_samples)
    tau_z_vector = np.random.uniform(low=0.050, high=1.050, size=n_samples)

    X = np.zeros((n_samples, 3))
    y = np.zeros((n_samples, 3))

    for index, (tau_z, max_w, min_w) in enumerate(zip(tau_z_vector, max_w_vector, min_w_vector)):
        dic = train_network(N, dt, training_time, inter_sequence_time, sequences, tau_z, tau_z_post,
                        tau_w, epochs=epochs, max_w=max_w, min_w=min_w, save_w_history=True, pre_rule=True)


        w = dic['w']
        self = w[pattern, pattern]
        exc = w[pattern, pattern_from]
        inh = w[pattern_from, pattern]
        y[index, 0] = self
        y[index, 1] = exc
        y[index, 2] = inh

        X[index, 0] = tau_z
        X[index, 1] = max_w
        X[index, 2] = min_w



    # Now we save both X and Y

    np.savez(outfile, x=X, y=y)
else:
    npz_file = np.load(outfile)
    X = npz_file['x']
    y = npz_file['y']

# Here we do the classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

reg = linear_model.LinearRegression(fit_intercept=False, n_jobs=-1)
reg.fit(X_train, y_train)
score = reg.score(X_test, y_test)
A = reg.coef_
B = np.linalg.inv(A)
print('score', score)

