#%%
import os,sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ML_stuff.dataset_tools import read_yaml_file
import time
import numpy as np
from types import SimpleNamespace
import matplotlib.pyplot as plt
from PO4AO.mbrl import get_env
import torch

args = SimpleNamespace(**read_yaml_file('../../Conf/papyrus_config.yaml'))


#%%
class ARPredictiveModel:
    def __init__(self, N, M, gamma=0.96, lam=1e-3):
        # N: Number of past WFS measurements
        # M: Number of future DM commands
        # gamma: Forgetting factor for the RLS algorithm
        
        self.N = N  # past horizon
        self.M = M  # future horizon
        self.gamma = gamma  # forgetting factor
        self.lam = lam  # regularization parameter

        self.y_past = np.zeros(N)  # past WFS measurements
        self.y_future = np.zeros(M)
        self.u_past = np.zeros(N)  # past DM commands
        self.u_future = np.zeros(M)  # future DM commands
        
        # Initialize AR coefficients A, B, and C (random initialization for demo)
        self.A = np.zeros((M, N))  # Correlates past WFS measurements to future
        self.B = np.zeros((M, N))  # Correlates past DM commands to future WFS
        self.C = np.zeros((M, M))  # Correlates future DM commands to future WFS
        
        self.Theta = np.concatenate([self.A, self.B, self.C], axis=1)
        # Initialize the inverse covariance matrix P
        self.P = np.eye(2*N + M) * 1e9 # Identity matrix with size according to past + future

    def predict_next_measurement(self, y_past, u_past, u_future):
        """ Predict the next WFS measurement using the AR model """
        # y_past: past WFS measurements, shape (N,)
        # u_past: past DM commands, shape (N,)
        # u_future: future DM commands, shape (M,)
        
        # Combine the inputs into one vector phi (concatenation of y_p, u_p, and u_f)
        phi = np.concatenate([y_past, u_past, u_future])
        
        # Prediction using the AR model: y_f = A * y_p + B * u_p + C * u_f
        y_pred = self.Theta @ phi
        
        return y_pred, phi


    def predict_cmd(self, y_past, u_past):
        """ Compute the optimal control signal u_f """
        # Regularized inversion term
        CTC_reg = self.C @ self.C.T + self.lam * np.eye(self.M)
        
        # Compute the inverse
        CTC_inv = np.linalg.inv(CTC_reg)
        
        # Create the concatenated matrix [A^T C, B^T C]
        ATC = self.A.T @ self.C
        BTC = self.B.T @ self.C

        
        # Combine the past WFS measurements and past DM commands
        y_u_concat = np.concatenate([y_past, u_past])

        print(CTC_inv.shape, ATC.shape, BTC.shape, y_u_concat.shape)
        print(np.concatenate([ATC, BTC], axis=0).T.shape)
        print((np.concatenate([ATC, BTC], axis=0).T@ y_u_concat).shape)
        
        # Compute control signal u_f
        u_f = - CTC_inv @ (np.concatenate([ATC, BTC], axis=0).T @ y_u_concat)
        
        return u_f


    def rls_update(self, y_true, y_past, u_past, u_future):
        """ Perform Recursive Least Squares (RLS) update """
        # y_true: true WFS measurement at step i+1
        # y_pred: predicted WFS measurement
        # phi: concatenated input vector used for prediction
        phi = np.concatenate([y_past, u_past, u_future])
        # Step 1: Compute the gain matrix K
        numerator = (1/self.gamma) * phi @ self.P
        gain_factor = 1 + (1/self.gamma) * phi@numerator
        K = numerator / gain_factor
        
        # Step 2: Calculate the prediction error
        y_pred = self.predict_next_measurement(y_past, u_past, u_future)[0]

        error = y_true - y_pred
        
        # Step 3: Update the AR model coefficients (Theta = A, B, C combined)
        self.Theta += np.outer(error,K.T)

        self.A = self.Theta[:, :self.N]
        self.B = self.Theta[:, self.N:self.N+self.M]
        self.C = self.Theta[:, self.N+self.M:]

        # Step 4: Update the inverse covariance matrix P
        self.P = (self.P - np.outer(K, phi @ self.P)) / self.gamma

    def fifo(self, array, new_value):
        """ Update an array in FIFO manner """
        np.roll(array, 1)
        array[0] = new_value
        return array


#%%

if __name__ == '__main__':

    # Define parameters
    N = 10  # number of past WFS measurements
    M = 3  # number of future DM commands


    # args.modulation = 3
    # env = get_env(args)
    # env.gainCL = 0.9

    arms = [ARPredictiveModel(N, M) for _ in range(env.dm.coefs.size)]

    y_pred = np.zeros((env.dm.coefs.size, M))
    y_true = np.zeros((env.dm.coefs.size, M))

    obs = env.reset_soft()
    action = np.zeros(env.dm.coefs.size)

    for i in range(args.nLoop):
        action =+ np.random.choice([-1,1], env.dm.coefs.size) * 1e-3

        obs,_, reward,strehl, done, info = env.step(i,torch.tensor(env.vec_to_img(action)))
        y_obs = env.img_to_vec(obs)

        print(strehl)

        obs = env.img_to_vec(obs)

        if i > 1:
            for j in range(env.dm.coefs.size):
                y_true[j] = arms[0].fifo(y_true[j], y_obs[j])

        for i in range(env.dm.coefs.size):
            arms[i].u_past = arms[i].fifo(arms[i].u_past, action[i])
            arms[i].y_past = arms[i].fifo(arms[i].y_past, obs[i])
            arms[i].u_future = arms[i].predict_cmd(arms[i].y_past, arms[i].u_past)
            arms[i].rls_update(y_true[i], arms[i].y_past, arms[i].u_past, arms[i].u_future)
        
        action = np.array([arms[i].u_future[-1] for i in range(env.dm.coefs.size)])
        

        



# %%
