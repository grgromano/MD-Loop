import numpy as np
import matplotlib.pyplot as plt

from toolbox.SINDy import SINDy
from toolbox.symbolic_SINDy import symbolic_SINDy
#from toolbox.cusum import CUSUM

from data.SINDy_data import evaluate_RMSE_d, evaluate_traj_d_1D

class Experiment():
    """
    :name: name of the product
    :price: price of the product
    :number: quantity of product in stock
    """
    def __init__(self, ode, ode_param, freq_SR, n_sample, noise_ratio, seed, n_seed, dt, dim_x, dim_k, SW_length, SW_length_SR, H, lazy, symbolic):
        
        # inputs:
        self.ode = ode
        self.ode_name = self.ode.name
        self.ode_param = ode_param
        self.freq_SR = freq_SR
        self.n_sample = n_sample
        self.noise_ratio = noise_ratio
        self.seed = seed
        self.n_seed = n_seed
        self.dt = dt
        self.dim_x = dim_x
        self.dim_k = dim_k
        self.SW_length = SW_length
        self.SW_length_SR = SW_length_SR
        self.H = H
        self.lazy = lazy
        self.symbolic = symbolic

        # internal variables:
        self.T0 = 0
        self.T = self.T0 + SW_length
        self.building_blocks = None 
        self.function_names = None
        self.model = None
        self.model_star = None

        self.rmse = 100
        self.model_complexity = 100
        self.lasso_penalty = 100
        self.patience = 0
        self.rmse_history = []
        # self.model_complexity_history = []
        # self.lasso_penalty_hystory = []
        self.model_history = [] 
        self.turning_points = [self.T0]


    def step_forward(self, SINDy_method, symbolic_SINDy_method, cusum, X_list_t, dX_list_t, param_list, feature_names):

        # t = 0 -> call SINDy:
        if self.T0 == 0:
            self.model, self.model_complexity, self.lasso_penalty = SINDy_method.call(X_list_t, dX_list_t, param_list, feature_names, self.dt)
            self.model_star = self.model
            self.model_history.append(self.model_star)
            self.turning_points.append(self.T)
            self.rmse, _ = evaluate_RMSE_d(self.model_star, self.ode, 10, 10, self.ode.init_high, self.ode.init_low, self.T-10, self.T, self.dim_k)
            change = cusum.update(self.rmse)

        else:

            # update CUSUM:
            change = cusum.update(self.rmse)
        
            if change or self.rmse > 0.15: # change-point detected  

                if self.symbolic: # -> call smart-SINDy:
                    self.model, self.building_blocks, self.function_names, model_complexity_SR, lasso_penalty_SR, self.patience = symbolic_SINDy_method.call(X_list_t, dX_list_t, param_list, feature_names, self.dt, self.building_blocks, self.function_names, self.patience, self.lazy, self.ode, self.ode_name, self.ode_param, self.freq_SR, self.n_sample, self.noise_ratio, self.seed, self.n_seed, self.T-self.SW_length_SR, self.T, self.dim_x, self.dim_k)
                    
                    if self.model is not None: # update model_star
                        self.model_complexity = model_complexity_SR
                        self.lasso_penalty = lasso_penalty_SR
                        self.model_star = self.model
                        self.model_history.append(self.model_star)
                        self.turning_points.append(self.T)
                
                else: # -> call SINDy again:
                    self.model, model_complexity_S, lasso_penalty_S = SINDy_method.call(X_list_t, dX_list_t, param_list, feature_names, self.dt)

                    if self.model is not None: # update model_star
                        self.model_complexity = model_complexity_S
                        self.lasso_penalty = lasso_penalty_S
                        self.model_star = self.model
                        self.model_history.append(self.model_star)
                        self.turning_points.append(self.T)
                
                cusum.reset()


        # evaluating the model:
        self.rmse, _ = evaluate_RMSE_d(self.model_star, self.ode, 10, 10, self.ode.init_high, self.ode.init_low, self.T-10, self.T, self.dim_k)
        print('RMSE: ', self.rmse) # RMSE on time window [T, T+10]
        print('')
        self.rmse_history.append(self.rmse)
        #self.model_complexity_history.append(self.model_complexity)
        #self.lasso_penalty_history.append(self.lasso_penalty)

        # advance time window:
        self.T0 += 1
        self.T += 1



    def plot(self, x_id=0):
        plot_times_1 = self.turning_points.copy() 
        plot_models_1 = self.model_history.copy()
        plot_times_1.append(self.H)
        plot_models_1.insert(0, plot_models_1[0])

        xt_true = []
        pred_list = []
        time_vector_1 = np.arange(0, plot_times_1[-1], self.dt)
        time_vector_2 = np.arange(plot_times_1[1], plot_times_1[-1], self.dt)
        for i in range(len(plot_models_1)):
            xt_true_i, pred_i = evaluate_traj_d_1D(plot_models_1[i], self.ode, 10, 1, [0.05, 0.05, 0], [0.05, 0.05, 0], plot_times_1[i], plot_times_1[i+1], x_id, self.dim_x, self.dim_k, plot=False)
            if i == 0:
                xt_true = np.concatenate((xt_true, xt_true_i), axis = 0)
            else: 
                xt_true = np.concatenate((xt_true, xt_true_i), axis = 0)
                pred_list = np.concatenate((pred_list, pred_i), axis = 0)
                
            
        # fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        # ax.plot(time_vector_2, pred_list, color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
        # ax.plot(time_vector_1, xt_true, color='red', linewidth=1.0, label='Correct Trajectory')
        # ax.scatter(time_vector_1[0], xt_true[0], color='green')
        # ax.scatter(time_vector_1[-1], xt_true[-1], color='red')
        # for x in [plot_times_1[1], plot_times_1[2]]: 
        #     ax.axvline(x=x, color='black', linestyle='--', linewidth=1.0, label='Vertical Line' if x == 1 else "")

        # ax.set_xlabel('t')
        # ax.set_ylabel('$x_{}(t)$'.format(x_id+1))
        # ax.set_xlim(0.-3, self.H+3)
        # ax.legend()
        # #ax.set_title("Symbolic SINDy on Oscillating Sel'kov Model")
        # ax.grid(True)

        # fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        # ax.plot(time_vector_2, pred_list, color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
        # ax.plot(time_vector_1, xt_true, color='red', linewidth=1.0, label='Correct Trajectory')
        # ax.scatter(time_vector_1[0], xt_true[0], color='green')
        # ax.scatter(time_vector_1[-1], xt_true[-1], color='red')
        # for x in [plot_times_1[1], plot_times_1[2]]: 
        #     ax.axvline(x=x, color='black', linestyle='--', linewidth=1.0, label='Vertical Line' if x == 1 else "")

        # ax.set_xlabel('t')
        # ax.set_ylabel('$x_{}(t)$'.format(x_id+1))
        # ax.set_xlim(0.-3, self.H+3)
        # ax.legend()
        # #ax.set_title("Symbolic SINDy on Oscillating Sel'kov Model")
        # ax.grid(True)



        time_vector_2 = time_vector_2[:450]
        pred_list = pred_list[:450]
        time_vector_1 = time_vector_1[:600]
        xt_true = xt_true[:600]


        print('plotting')
        print(np.shape(time_vector_2))
        print(np.shape(pred_list))
        print(np.shape(time_vector_1))
        print(np.shape(xt_true))

        # print('plotting')
        # print(time_vector_2)
        # print(pred_list)
        # print(time_vector_1)
        # print(xt_true)

        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(time_vector_2, pred_list, color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
        ax.plot(time_vector_1, xt_true, color='red', linewidth=1.0, label='Correct Trajectory')
        ax.scatter(time_vector_1[0], xt_true[0], color='green')
        ax.scatter(time_vector_1[-1], xt_true[-1], color='red')
        for x in [plot_times_1[1], plot_times_1[2]]: 
            ax.axvline(x=x, color='black', linestyle='--', linewidth=1.0, label='Vertical Line' if x == 1 else "")

        ax.set_xlabel('t')
        ax.set_ylabel('$x_{}(t)$'.format(x_id+1))
        ax.set_xlim(0.-1, 60+1)
        ax.legend()
        #ax.set_title("Symbolic SINDy on Oscillating Sel'kov Model")
        ax.grid(True)


    def plot_RMSE(self):
        time_vector = np.arange(self.turning_points[1], self.H+1)
        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        ax.plot(time_vector, self.rmse_history, 'ro')
        ax.plot(time_vector, self.rmse_history, 'r-', linewidth=0.3)
        for x in [self.turning_points[1], self.turning_points[2]]:  
            ax.axvline(x=x, color='black', linestyle='--', linewidth=1.0, label='Vertical Line' if x == 1 else "")
        ax.set_xlabel('t')
        ax.set_ylabel('RMSE')
        ax.set_xlim(0.-3, self.H+3)
        #ax.set_title('RMSE time-series')

            