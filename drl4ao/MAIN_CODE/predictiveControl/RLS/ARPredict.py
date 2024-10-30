import numpy as np

class ARPredictiveModel:
    def __init__(self, N, M, gamma=0.96, lam=1e-3):
        # N: Number of past WFS measurements
        # M: Number of future DM commands
        # gamma: Forgetting factor for the RLS algorithm
        
        self.N = N  # past horizon
        self.M = M  # future horizon
        self.gamma = gamma  # forgetting factor
        self.lam = lam  # regularization parameter
        
        # Initialize AR coefficients A, B, and C (random initialization for demo)
        self.A = np.random.randn(M, N)  # Correlates past WFS measurements to future
        self.B = np.random.randn(M, N)  # Correlates past DM commands to future WFS
        self.C = np.random.randn(M, M)  # Correlates future DM commands to future WFS
        
        # Initialize the inverse covariance matrix P
        self.P = np.eye(N + M + M)  # Identity matrix with size according to past + future

    def predict_next_measurement(self, y_past, u_past, u_future):
        """ Predict the next WFS measurement using the AR model """
        # y_past: past WFS measurements, shape (N,)
        # u_past: past DM commands, shape (N,)
        # u_future: future DM commands, shape (M,)
        
        # Combine the inputs into one vector phi (concatenation of y_p, u_p, and u_f)
        phi = np.concatenate([y_past, u_past, u_future])
        
        # Prediction using the AR model: y_f = A * y_p + B * u_p + C * u_f
        y_pred = np.dot(self.A, y_past) + np.dot(self.B, u_past) + np.dot(self.C, u_future)
        
        return y_pred, phi


    def compute_control_signal(self, y_past, u_past):
        """ Compute the optimal control signal u_f """
        # Regularized inversion term
        CTC_reg = np.dot(self.C.T, self.C) + self.lam * np.eye(self.M)
        
        # Compute the inverse
        CTC_inv = np.linalg.inv(CTC_reg)
        
        # Create the concatenated matrix [A^T C, B^T C]
        ATC = np.dot(self.A.T, self.C)
        BTC = np.dot(self.B.T, self.C)
        
        # Combine the past WFS measurements and past DM commands
        y_u_concat = np.concatenate([y_past, u_past])
        
        # Compute control signal u_f
        u_f = -np.dot(CTC_inv, np.dot(np.hstack([ATC, BTC]), y_u_concat))
        
        return u_f


    def rls_update(self, y_true, y_pred, phi):
        """ Perform Recursive Least Squares (RLS) update """
        # y_true: true WFS measurement at step i+1
        # y_pred: predicted WFS measurement
        # phi: concatenated input vector used for prediction
        
        # Step 1: Compute the gain matrix K
        phi_T_P = np.dot(phi.T, self.P)
        gain_factor = 1 + (self.gamma)**(-1) * np.dot(phi_T_P, phi)
        K = ((self.gamma)**(-1) * phi_T_P) / gain_factor
        
        # Step 2: Calculate the prediction error
        error = y_true - y_pred
        
        # Step 3: Update the AR model coefficients (Theta = A, B, C combined)
        Theta = np.concatenate([self.A.flatten(), self.B.flatten(), self.C.flatten()])
        Theta = Theta + K * error
        
        # Reshape Theta back into A, B, C
        split1 = self.A.size
        split2 = split1 + self.B.size
        self.A = Theta[:split1].reshape(self.A.shape)
        self.B = Theta[split1:split2].reshape(self.B.shape)
        self.C = Theta[split2:].reshape(self.C.shape)
        
        # Step 4: Update the inverse covariance matrix P
        self.P = (self.P - np.outer(K, phi_T_P)) / self.gamma


if __name__ == '__main__':

    # Define parameters
    N = 5  # number of past WFS measurements
    M = 3  # number of future DM commands
    gamma = 0.95  # forgetting factor
    

    # Initialize the AR model
    ar_model = ARPredictiveModel(N, M, gamma)

    # Example past and future data
    y_past = np.random.randn(N)  # past WFS measurements
    u_past = np.random.randn(N)  # past DM commands
    u_future = np.random.randn(M)  # future DM commands

    # True future WFS measurement (for training)
    y_true = np.random.randn(M)  # this would come from actual data

    # Prediction
    y_pred, phi = ar_model.predict_next_measurement(y_past, u_past, u_future)

    # Perform RLS update with true measurement
    ar_model.rls_update(y_true, y_pred, phi)
