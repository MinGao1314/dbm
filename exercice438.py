import plotly.graph_objs as go
import plotly.offline
import numpy as np


class DPP438:
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
        real_values = np.random.uniform(-0.05, 0.05, size=(self.n_traj, M))
        im_values = np.random.uniform(-0.05, 0.05, size=(self.n_traj, M))
        A = np.matrix(real_values + 1j * im_values)
        V_0 = np.dot(A , A.H)  # le .H conjugue puis transpose
        self.dpp_matrix[0] = V_0
        return V_0

    def generate(self):
        for sample in range(self.n_samples-1):
            real_values = np.random.randn(self.n_traj, self.n_traj)
            im_values = np.random.randn(self.n_traj, self.n_traj)
            A = np.matrix(real_values + 1j * im_values) * (1/2)**0.5
            V_t = np.dot(A , A.H)
            self.dpp_matrix[sample+1] = self.dpp_matrix[sample] + (self.dt**(1/2) * V_t)

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
        test = DPP438(100, 10, 1, 3)
        test.plot('test438.html')