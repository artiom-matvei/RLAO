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

args = SimpleNamespace(**read_yaml_file('../../Conf/papyrus_config.yaml'))


#%%

# env.atm.generateNewPhaseScreen(17)
class DMDc:
    def __init__(self, N, M, env):
        # N: Number of past WFS measurements
        # M: Number of modes to control
        # env: AO environment object
        self.N = N  # past horizon
        self.M = M  # number of modes to control
        self.env = env  # AO environment object
        self.burn_in = 100  # number of iterations to burn in the model
        self.scale = 1e6 # scaling factor for the WFS measurements
        self.C2M = np.linalg.pinv(self.env.M2C_CL.copy())

        # State space model
        self.B = np.eye(self.M)

    def get_obs_matrices(self):
        X0 = np.zeros((self.M, self.N))
        X1 = np.zeros((self.M, self.N + 1))
        C  = np.zeros((self.M, self.N))

        self.env.atm.generateNewPhaseScreen(np.random.randint(0, 1000))
        obs = self.env.reset_soft()

        for i in range(self.N + self.burn_in + 1):
            action = self.env.gainCL * obs
            obs,_, reward,strehl, done, info = self.env.step(i,action)

            if i >= self.burn_in and i != self.N + self.burn_in:
                obs_now = np.matmul(self.env.reconstructor, self.env.wfsSignal)
                modal_obs_pol = self.C2M@(obs_now + self.env.dm.coefs.copy()) * self.scale
                modal_obs_cl = self.C2M@obs_now * self.scale
                X0[:, i - self.burn_in] = modal_obs_pol[:self.M] 
                X1[:, i - self.burn_in] = modal_obs_pol[:self.M] 
                C[:, i - self.burn_in] = self.env.gainCL * self.control_signal(modal_obs_cl[:self.M])
            elif i == self.N + self.burn_in:
                obs_now = np.matmul(self.env.reconstructor, self.env.wfsSignal) * self.scale
                modal_obs_pol = self.C2M@(obs_now + self.env.dm.coefs.copy())
                X1[:, i - self.burn_in] = modal_obs_pol[:self.M] 

        return X0, X1[:,1:], C


    def control_signal(self, obs):
        return - obs
    

    def construct_transition_matrix(self, X0, X1, C, do_B=True):
        if not do_B:
            # Construct the transition matrix A
            # Truncated SVD of X0 with truncation rank r
            U, S, Vt = np.linalg.svd(X0, full_matrices=False)
            # r = np.linalg.matrix_rank(X0)
            # U_r = U[:, :r]
            # S_r = np.diag(S[:r])
            # Vt_r = Vt[:r, :]
            print(U.shape, S.shape, Vt.shape)
            rank = np.sum(S > 1e-10)  # Count non-zero singular values
            print("Rank:", rank)
            # Construct the matrix A
            A = np.matmul(X1 - np.matmul(self.B, C), np.matmul(Vt.T, np.matmul(np.linalg.inv(np.diag(S)), U.T)))

            self.A = A
            return A
        else:
                Omega = np.vstack((X0, C))

                U_tilde, Sigma_tilde, V_tilde_conj = np.linalg.svd(Omega, full_matrices=False)

                n = X0.shape[0]
                U_tilde_1 = U_tilde[:n, :]
                U_tilde_2 = U_tilde[n:, :]

                A_approx = X1 @ V_tilde_conj.T @ np.linalg.inv(np.diag(Sigma_tilde)) @ U_tilde_1.T
                B_approx = X1 @ V_tilde_conj.T @ np.linalg.inv(np.diag(Sigma_tilde)) @ U_tilde_2.T

                self.B = B_approx
                self.A = A_approx

                return A_approx, B_approx


    def predict(self, obs, cmd):
        # Construct the transition matrix A
        pred_obs = np.matmul(self.A, obs) + np.matmul(self.B, cmd)
        return pred_obs
    
    def test_predict(self, len):
        # Construct the transition matrix A
        obs_hist = np.zeros((self.M, len))
        next_obs = np.zeros((self.M, len + 1))
        pred = np.zeros((self.M, len))

        self.env.atm.generateNewPhaseScreen(np.random.randint(0, 1000))
        obs = self.env.reset_soft()

        for i in range(len + self.burn_in + 1):
            action = self.env.gainCL * obs
            obs,_, reward,strehl, done, info = self.env.step(i,action)

            if i >= self.burn_in and i != len + self.burn_in:
                obs_now = np.matmul(self.env.reconstructor, self.env.wfsSignal)
                modal_obs_pol = self.C2M@(obs_now + self.env.dm.coefs.copy()) * self.scale
                modal_obs_cl = self.C2M@obs_now * self.scale
                obs_hist[:, i - self.burn_in] = modal_obs_pol[:self.M]
                next_obs[:, i - self.burn_in] = modal_obs_pol[:self.M]
                cmd_now = env.gainCL * self.control_signal(modal_obs_cl[:self.M])
                pred[:, i - self.burn_in] = self.predict(modal_obs_pol[:self.M], cmd_now)
            elif i == len + self.burn_in:
                obs_now = np.matmul(self.env.reconstructor, self.env.wfsSignal)
                modal_obs_pol = self.C2M@(obs_now + self.env.dm.coefs.copy()) * self.scale
                next_obs[:, i - self.burn_in] = modal_obs_pol[:self.M]

        return obs_hist, next_obs[:, 1:], pred

#%%


if __name__ == '__main__':


    args.modulation = 3
    env = get_env(args)
    env.gainCL = 0.9

    # Define parameters
    N = 5000  # number of past WFS measurements
    M = 10  # number of modes to predict

    dmd = DMDc(N, M, env)

    dmd.env.tel.resetOPD()
    dmd.env.dm.coefs = 0
    dmd.env.tel*dmd.env.dm*dmd.env.wfs

    print("Starting data sequence")
    x0, x1, c = dmd.get_obs_matrices()
    print("Starting model computation")

    A, B = dmd.construct_transition_matrix(x0, x1, c, do_B=True)
    print("Starting testing sequence")

    dmd.env.tel.resetOPD()
    dmd.env.dm.coefs = 0
    dmd.env.tel*dmd.env.dm*dmd.env.wfs

    o,n,p = dmd.test_predict(100)


    import matplotlib.gridspec as gridspec
    plt.style.use('ggplot')
    cmap = plt.get_cmap('inferno')
    num_modes = 10

    st = ["No","DMDc"]

    for i,vals in enumerate([o,p]):

        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        ax1 = plt.subplot(gs[0])
        custom_colors = []
        for mode in range(num_modes):
            bins = np.arange(-0.02, 0.02 + 0.001, 0.001)
            # counts, bins = np.histogram(pred[:-delay,mode] - modes[n + delay - 1:, mode], bins=bins)
            # # Plot the outline using plt.step()
            color = cmap(1 - (mode / (num_modes)))
            custom_colors.append(color)
            # plt.step(bins[:-1], counts, where='mid', color=color, label=f'Mode #{mode + 1}')
            # plt.vlines(bins[0], 0, counts[0], colors=color)  # Leftmost vertical bar
            # plt.vlines(bins[-2], 0, counts[-1], colors=color)
            ax1.hist(n[mode, :] - vals[mode, :], color=color, bins=bins, alpha=np.linspace(1, 0.4, num_modes)[mode], label=f'Mode #{mode + 1}')
            ax1.set_ylim(0, 30)

        ax1.set_title(f'Histogram of Residuals - {st[i]} Prediction')

        ax2 = plt.subplot(gs[1])  # Second subplot in the grid (smaller)
        ax2.scatter(np.arange(1, num_modes + 1), np.std(vals - n, axis=1) , color=custom_colors, s=70, alpha=0.8)
        ax2.set_title('Standard Deviation of Residuals per Mode')
        ax2.set_xlabel('Mode Number')
        ax2.set_ylim(0, 0.015)
        # ax2.set_yscale('log')
        # plt.title('Residual values from prediction')
        ax1.legend()
        plt.show()



# %%
