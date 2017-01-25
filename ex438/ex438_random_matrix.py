import plotly.graph_objs as go
import plotly.offline
import numpy as np


class DPP438_random_matrix:
    def __init__(self, n_traj, n_samples, tf, M):
        self.n_traj = n_traj
        self.n_eigen_values = M
        self.n_samples = n_samples
        self.T = tf
        self.dt = tf/n_samples
        self.dpp_matrix = np.zeros((self.n_samples, self.n_traj, self.n_eigen_values))
        self.initialisation()
        self.generate()
        self.eigen_values = np.zeros((self.n_samples, self.n_eigen_values))
        self.diag()

    def initialisation(self):
        real_values = np.random.uniform(-0.05, 0.05, size=(self.n_traj, self.n_eigen_values))
        im_values = np.random.uniform(-0.05, 0.05, size=(self.n_traj, self.n_eigen_values))
        V_0 = np.matrix(real_values + 1j * im_values)
        self.dpp_matrix[0] = V_0
        return V_0

    def generate(self):
        for sample in range(self.n_samples-1):
            real_values = np.random.randn(self.n_traj, self.n_eigen_values)
            im_values = np.random.randn(self.n_traj, self.n_eigen_values)
            V_t = np.matrix(real_values + 1j * im_values) * (1/2)**0.5
            self.dpp_matrix[sample+1] = self.dpp_matrix[sample] + (self.dt**(1/2) * V_t)

    def diag(self):
        for sample in range(self.n_samples):
            V_t_star_V_t = np.dot(np.matrix(self.dpp_matrix[sample]).H, self.dpp_matrix[sample])
            self.eigen_values[sample] = sorted(np.linalg.eigvals(V_t_star_V_t), reverse=True)

    def plot(self, filename):
        data=[]
        for traj in range(self.n_eigen_values):
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
        test = DPP438_random_matrix(30, 100, 1, 25)
        test.plot('test438_random_matrix.html')