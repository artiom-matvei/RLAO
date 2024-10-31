import numpy as np

class LEPredictiveModel:
    def __init__(self, M=1, modal=False, num_modes=300, damping=-0.42):
        # N: Number of past WFS measurements
        # M: Number of future DM commands
        # gamma: Forgetting factor for the RLS algorithm

        self.M = M  # future horizon
        self.modal = modal

        if self.modal:
            self.num_modes = num_modes
            self.damping = damping
            self.alpha = np.arange(1, num_modes + 1)**damping
        

    def predict_next_command(self, obs_buffer, prev_command):
        """ Predict the next command using the LE model """
        # obs_buffer: past measurements, shape (2,)
        # prev_command: previous DM command, shape (1,)

        a_t = prev_command + obs_buffer[0] - obs_buffer[1]

        if self.modal:
            a_t = self.alpha * a_t

        
        return a_t

