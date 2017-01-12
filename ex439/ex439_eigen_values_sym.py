import plotly.graph_objs as go
import plotly.offline
import numpy as np


class DPP439_eigen_values:
    def __init__(self, n_traj, n_samples, tf):
        self.n_traj = n_traj
        self.n_samples = n_samples
        self.T = tf
        self.dt = tf/n_samples
        self.eigen_values = np.zeros((self.n_samples, self.n_traj))
        self.initialisation()
        self.generate()

    def initialisation(self):
        #real_values = np.random.uniform(-0.05, 0.05, size=(self.n_traj, self.n_traj))
        #im_values = np.random.uniform(-0.05, 0.05, size=(self.n_traj, self.n_traj))
        #V_0 = np.matrix(real_values + 1j * im_values)
        #self.eigen_values[0] = sorted(np.real(np.linalg.eigvals(V_0)), reverse=False)
        self.eigen_values[0] = np.linspace(1, self.n_traj/5, num=self.n_traj)
        #self.eigen_values[0][1]=self.eigen_values[0][2]+0.1
        return self.eigen_values

    def generate(self):
        for sample in range(self.n_samples-1):
            #print(self.eigen_values[sample])
            for i in range(self.n_traj):
                lbda_i = self.eigen_values[sample][i]
                eigen_values_list = [lbda for lbda in list(self.eigen_values[sample]) if lbda != lbda_i]
                #print(lbda_i, "\n", eigen_values_list)
                sum_term = sum ([ (1/(lbda_i - lbda_k)) for lbda_k in eigen_values_list ])
                #print("sumterm", sum_term)
                W = (self.dt)**(0.5) * np.random.randn()
                self.eigen_values[sample+1][i] = self.eigen_values[sample][i] + \
                                                W *(2/self.n_traj)**(0.5) + \
                                                 (1/self.n_traj)*sum_term * self.dt - \
                                                 self.eigen_values[sample][i] * self.dt
                #print("eigen:", self.eigen_values[sample+1][i])

        return self.eigen_values

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
    test = DPP439_eigen_values(10, 1000, 1)
    #print(test.eigen_values)
    test.plot('test439_eigen_values.html')