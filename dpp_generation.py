import plotly.graph_objs as go
import plotly.offline
from utilitary import *


class DPP:
    def __init__(self, n_traj, n_samples, tf):
        self.n_traj = n_traj
        self.n_samples = n_samples
        self.T = tf
        self.dt = tf/n_samples
        self.dpp_matrix = np.zeros((self.n_samples, self.n_traj, self.n_traj))
        self.generate()
        self.eigen_values = np.zeros((self.n_samples, self.n_traj))
        self.diag()

    def generate(self):
        self.dpp_matrix[0] = real_hermitian_matrix_generation(self.n_traj)
        for sample in range(self.n_samples-1):
            self.dpp_matrix[sample+1] = self.dpp_matrix[sample] + (self.dt**(1/2) * real_GUE(self.n_traj))

    def diag(self):
        for sample in range(self.n_samples):
            self.eigen_values[sample] = np.linalg.eigvals(self.dpp_matrix[sample])

    def plot(self, filename):
        data=[]
        for traj in range(self.n_traj):
            traj_trace = go.Scatter(
                x=np.array(range(self.n_samples)),
                y=(self.eigen_values.T)[traj],
                mode='lines+markers')
            data.append(traj_trace)
        plotly.offline.init_notebook_mode()
        plotly.offline.plot(data, filename=''.join(['plot/', filename]))


if __name__ == '__main__':
    print(DPP(10,10,1).plot('test'))



