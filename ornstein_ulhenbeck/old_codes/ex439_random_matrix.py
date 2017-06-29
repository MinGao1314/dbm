import plotly.graph_objs as go
import plotly.offline
import numpy as np
from utilitary import herm_matrix


class DPP439_random_matrix:
    def __init__(self, n_traj, n_samples, tf, M):
        self.n_traj = n_traj
        self.n_samples = n_samples
        self.T = tf
        self.dt = tf/n_samples
        self.dpp_matrix = np.zeros((self.n_samples, self.n_traj, self.n_traj))
        self.initialisation(M)
        self.generate()
        self.eigen_values = np.zeros((self.n_samples, self.n_traj))
        self.diag()


    def initialisation(self, M):
        real_values = np.random.uniform(-0.0005, 0.0005, size=(self.n_traj, self.n_traj))
        im_values = np.random.uniform(-0.0005, 0.0005, size=(self.n_traj, self.n_traj))
        V_0 = np.matrix(real_values + 1j * im_values)
        self.dpp_matrix[0] = V_0
        return V_0

    def generate(self):
        for sample in range(self.n_samples-1):
            self.dpp_matrix[sample+1] = self.dpp_matrix[sample] + herm_matrix(self.n_traj, beta=2)*(self.dt)**0.5 - self.dpp_matrix[sample]*self.dt

    def diag(self):
        for sample in range(self.n_samples):
            self.eigen_values[sample] = sorted(np.real(np.linalg.eigvals(self.dpp_matrix[sample])), reverse=True)

    def plot(self, filename):
        data=[]
        for traj in range(self.n_traj):
            traj_trace = go.Scatter(
                x=self.dt*np.array(range(self.n_samples)),
                y=(self.eigen_values.T)[traj],
                mode='lines')
            data.append(traj_trace)
        layout = go.Layout(showlegend=False)
        fig = go.Figure(data=data, layout=layout)
        plotly.offline.init_notebook_mode()
        plotly.offline.plot(fig, filename=''.join(['plot/', filename]))

if __name__ == '__main__':
        test = DPP439_random_matrix(50, 100, 1, 3)
        test.plot('test439_random_matrix.html')
