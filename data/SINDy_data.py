#from Data import *
import numpy as np
import data.equations as equations
import data.data as data
from data.interpolate import num_diff
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def SINDy_data(ode_name, ode_param, freq, n_sample, noise_ratio, dim_x, dim_k, T0=0, T=15):

    np.random.seed(999)
    alg = 'tv'

    ode = equations.get_ode(ode_name, ode_param)
    T = ode.T # !!!!!!!!!!! se si vuole fare scorrevole, commentare questa riga 
    init_low = ode.init_low 
    init_high = ode.init_high 
    has_coef = ode.has_coef 
    noise_sigma = ode.std_base * noise_ratio

    # data simulation: 
    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high) 
    yt = dg.generate_data()
    #print(np.shape(yt)) 
    # print(yt[0:20, 0, 0]) 
    # print(yt[0:20, 0, 1]) 
    # print(yt[0:20, 0, 2])

    if T0: # if T0>0, cut portion [0,T0] 
        yt = yt[T0*freq:, :, :]
    # print('Dataset shape: ', np.shape(yt))
    # print(yt[0:20, 0, 0]) 
    # print(yt[0:20, 0, 1])
    # print(yt[0:20, 0, 2])

    # numerical differentiation:
    if noise_sigma == 0:
        dxdt_hat = (yt[1:, :, :] - yt[:-1, :, :]) / (dg.solver.t[1:] - dg.solver.t[:-1])[:, None, None]
    else:
        dxdt_hat = num_diff(yt, dg, alg, T0)
    #print(np.shape(dxdt_hat))
    #print('Numerical differentiation: Done.')


    # build dataset:
    X_train = yt[:-1, :, :dim_x] # yt[:-1, :, :-1]
    #print(np.shape(X_train))
    X_train = np.transpose(X_train, (1, 0, 2))
    #print(np.shape(X_train))
    dX_train = dxdt_hat[:, :, :dim_x] # dxdt_hat[:, :, :-1]
    #print(np.shape(dX_train))
    dX_train = np.transpose(dX_train, (1, 0, 2))
    #print(np.shape(dX_train))

    if dim_k != 0:
        param_train = yt[:-1, :, -dim_k:] # yt[:-1, :, -1]
        param_train = np.transpose(param_train, (1, 0, 2))
        param_train = param_train.squeeze()

    X_list = []
    dX_list = []
    param_list = []
    n_train = len(X_train)
    for i in range(n_train):
        X_list.append(X_train[i])
        dX_list.append(dX_train[i])
        if dim_k != 0:
            param_list.append(param_train[i])
    
    if dim_x == 1:
        feature_names = ["X0"]
        if dim_k == 1:
            feature_names += ["X1"]
        elif dim_k == 2:
            feature_names += ["X1", "X2"]
    elif dim_x == 2:
        feature_names = ["X0", "X1"]
        if dim_k == 1:
            feature_names += ["X2"]
    else: # dim_x == 3
        feature_names = ["X0", "X1", "X2"]
        if dim_k == 1:
            feature_names += ["X3"]

    # if dim_k == 1:
    #     feature_names +=  ["a"]
    # elif dim_k == 2:
    #     feature_names +=  ["a", "b"]

    return X_list, dX_list, param_list, feature_names



def SINDy_data_HD(ode_name, ode_param, freq, n_sample, noise_ratio, dim_x, dim_k, T0=0, T=15, latent_data = None):

    np.random.seed(999)
    alg = 'tv'

    ode = equations.get_ode(ode_name, ode_param)
    #T = ode.T
    init_low = ode.init_low 
    init_high = ode.init_high 
    has_coef = ode.has_coef 
    noise_sigma = ode.std_base * noise_ratio

    # data simulation: 
    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high) 
    #yt = dg.generate_data()
    yt = latent_data
    #print(np.shape(yt)) 
    # print(yt[0:20, 0, 0]) 
    # print(yt[0:20, 0, 1]) 
    # print(yt[0:20, 0, 2])

    if T0: # if T0>0, cut portion [0,T0] 
        yt = yt[T0*freq:, :, :]
    # print('Dataset shape: ', np.shape(yt))
    # print(yt[0:20, 0, 0]) 
    # print(yt[0:20, 0, 1])
    # print(yt[0:20, 0, 2])

    # numerical differentiation:
    value = 5 / 150
    array = np.full((150, 1, 1), value)
    #print((dg.solver.t[1:] - dg.solver.t[:-1])[:, None, None])
    #print(np.shape((dg.solver.t[1:] - dg.solver.t[:-1])[:, None, None]))
    #print(array)
    #print(np.shape(array))
    if noise_sigma == 0:
        #dxdt_hat = (yt[1:, :, :] - yt[:-1, :, :]) / (dg.solver.t[1:] - dg.solver.t[:-1])[:, None, None]
        dxdt_hat = (yt[1:, :, :] - yt[:-1, :, :]) / array # TODO: non so perché, ma la versione precedente utilizzava un dt sbagliato, qui ho manualmente impostato dt = 5/150, da rendere generico
    else:
        dxdt_hat = num_diff(yt, dg, alg, T0)
    #print(np.shape(dxdt_hat))
    #print('Numerical differentiation: Done.')


    # build dataset:
    X_train = yt[:-1, :, :dim_x] # yt[:-1, :, :-1]
    #print(np.shape(X_train))
    X_train = np.transpose(X_train, (1, 0, 2))
    #print(np.shape(X_train))
    dX_train = dxdt_hat[:, :, :dim_x] # dxdt_hat[:, :, :-1]
    #print(np.shape(dX_train))
    dX_train = np.transpose(dX_train, (1, 0, 2))
    #print(np.shape(dX_train))

    if dim_k != 0:
        param_train = yt[:-1, :, -dim_k:] # yt[:-1, :, -1]
        param_train = np.transpose(param_train, (1, 0, 2))
        param_train = param_train.squeeze()

    X_list = []
    dX_list = []
    param_list = []
    n_train = len(X_train)
    for i in range(n_train):
        X_list.append(X_train[i])
        dX_list.append(dX_train[i])
        if dim_k != 0:
            param_list.append(param_train[i])
    
    if dim_x == 1:
        feature_names = ["X0"]
        if dim_k == 1:
            feature_names += ["X1"]
        elif dim_k == 2:
            feature_names += ["X1", "X2"]
    elif dim_x == 2:
        feature_names = ["X0", "X1"]
        if dim_k == 1:
            feature_names += ["X2"]
    else: # dim_x == 3
        feature_names = ["X0", "X1", "X2"]
        if dim_k == 1:
            feature_names += ["X3"]

    # if dim_k == 1:
    #     feature_names +=  ["a"]
    # elif dim_k == 2:
    #     feature_names +=  ["a", "b"]

    return X_list, dX_list, param_list, feature_names




# TODO: fix to deal with dim_x = 1
def existence_conditions(X_list, init_low, n_variables):
    # ensure existance conditions:

    if n_variables == 1:
        lb = init_low + 0.001
        for i in range(np.shape(X_list)[0]):
            for j in range(np.shape(X_list)[1]):
                if X_list[i][j] < 1e-4:
                    X_list[i][j] = lb
    else:
        for idx in range(n_variables):
            lb = init_low[idx] + 0.001
            for i in range(np.shape(X_list)[0]):
                for j in range(np.shape(X_list)[1]):
                    if X_list[i][j][idx] < 0:
                        X_list[i][j][idx] = lb
    return X_list
#just in case:
# lb = ode.init_low[dim_x + idx] + 0.001
# for i in range(np.shape(param_list)[0]):
#     for j in range(np.shape(param_list)[1]):
#         if param_list[i][j] < 0:
#             param_list[i][j] = lb
# for idx in range(dim_k):
#     lb = ode.init_low[dim_x + idx] + 0.001
#     for i in range(np.shape(param_list)[0]):
#         for j in range(np.shape(param_list)[1]):
#             if param_list[i][j][idx] < 0:
#                 param_list[i][j][idx] = lb



def evaluate_RMSE(model, ode, freq, n_sample, init_high, init_low, dim_k=1):
    # function computing the RMSE (and the MSE) of a given model: 

    np.random.seed(666)
    dt = 1 / freq
    TIME = min(ode.T, 100) 

    # true trajectories:
    dg_true = data.DataGenerator(ode, TIME, freq, n_sample, noise_sigma=0., init_high=init_high, init_low=init_low)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    # estimated trajectories:
    pred_0_list = []
    for i in range(n_sample):

        correct_param = xt_true[0, i, -dim_k:]
        #print(np.shape(correct_param))

        t = np.arange(0,TIME,dt)
        T_plot = len(t)
        test_params = np.tile(correct_param, (T_plot,1))
        if dim_k != 0:
            pred_0 = model.simulate(xt_true[0, i, :][:-dim_k], t= t[:T_plot], u = test_params)
        else: # dim_k == 0
            pred_0 = model.simulate(xt_true[0, i, :], t= t[:T_plot])
        pred_0_list.append(pred_0)
    #print(np.shape(pred_0_list))


    xt_true = xt_true[:len(pred_0), :, :]
    if dim_k != 0:
        xt_true = xt_true[:, :, :-dim_k]
    xt_true = xt_true.squeeze()
    #print(np.shape(xt_true)) #(151, 25, 2)
    pred_0_list = np.transpose(pred_0_list, (1, 0, 2))
    pred_0_list = pred_0_list.squeeze()
    #print(np.shape(pred_0_list)) #(151, 25, 2)

    # RMSE:
    rmse_0_list = []
    mse_0_list = []
    for i in range(n_sample):
        rmse_0 = root_mean_squared_error(xt_true[:, i], pred_0_list[:,i]) 
        mse_0 = mean_squared_error(xt_true[:, i], pred_0_list[:,i])
        rmse_0_list.append(rmse_0)
        mse_0_list.append(mse_0)
    rmse_0 = np.mean(rmse_0_list)
    mse_0 = np.mean(mse_0_list)
    return rmse_0, mse_0



def evaluate_traj(model, ode, freq, n_sample, init_high, init_low, dim_x=1, dim_k=1, title=None, T_aux=100):
    # function plotting the true and estimated trajectories:# function displaying the estimated trajectories:

    assert n_sample in [1, 4], "n_sample should be 1 or 4."

    T_aux = min(ode.T, T_aux)

    # print('dim_x:', dim_x)
    # print('dim_k:', dim_k)

    np.random.seed(666)
    dt = 1 / freq
    time_vector = np.arange(0, T_aux + dt, dt)
    t = np.arange(0,T_aux,dt)
    T_plot = len(t)

    # true trajectories:
    dg_true = data.DataGenerator(ode, T_aux, freq, n_sample, noise_sigma=0., init_high=init_high, init_low=init_low)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    # estimated trajectories:
    pred_0_list = []
    for i in range(n_sample):
        correct_param = xt_true[0, i, -dim_k:]
        #print(correct_param)
        t = np.arange(0,T_aux,dt)
        T_plot = len(t)
        test_params = np.tile(correct_param, (T_plot,1))
        if dim_k != 0:
            pred_0 = model.simulate(xt_true[0, i, :][:-dim_k], t= t[:T_plot], u = test_params)
        else:
            pred_0 = model.simulate(xt_true[0, i, :], t= t[:T_plot])
        pred_0_list.append(pred_0)
    #print(np.shape(pred_0_list))

    xt_true = xt_true[:len(pred_0), :, :]
    #print(np.shape(xt_true))
    time_vector = time_vector[:len(pred_0)]
    if dim_k != 0: # RMK!!! nel caso dim_k=0, xt_true = xt_true[:, :, :-dim_k] elimina la colonna -1, con output avente shape: (300, 1, 0) anziché (300, 1, 2)
        xt_true = xt_true[:, :, :-dim_k]
    #print(np.shape(xt_true))
    xt_true = xt_true.squeeze()
    #print(np.shape(xt_true))
    pred_0_list = np.transpose(pred_0_list, (1, 0, 2))
    pred_0_list = pred_0_list.squeeze()
    #print(np.shape(pred_0_list))

    if n_sample == 1:
        if dim_x == 1:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.plot(time_vector, pred_0_list, color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
            #ax.plot(time_vector, xt_true, color='red', linewidth=1.0, label='Correct Trajectory')
            ax.plot(time_vector, xt_true, color='red', linewidth=1.0, label='Correct Trajectory', linestyle='--')
            ax.scatter(time_vector[0], xt_true[0], color='green', label='Start')
            ax.scatter(time_vector[-1], xt_true[-1], color='red', label='End')
            if title:
                ax.set_title(title)
            ax.set_xlabel('$t$')
            ax.legend()
            ax.grid(True)
        elif dim_x == 2:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.plot(pred_0_list[:, 0], pred_0_list[:, 1], color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
            ax.plot(xt_true[:, 0], xt_true[:, 1], color='red', linewidth=1.0, label='Correct Trajectory')
            ax.scatter(xt_true[:, 0][0], xt_true[:, 1][0], color='green', label='Start')
            ax.scatter(xt_true[:, 0][-1], xt_true[:, 1][-1], color='red', label='End')
            if title:
                ax.set_title(title)
            ax.legend()
            ax.grid(True) 
        else: # dim_x == 3
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
            ax.plot(pred_0_list[:, 0], pred_0_list[:, 1], pred_0_list[:, 2], color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
            ax.plot(xt_true[:, 0], xt_true[:, 1], xt_true[:, 2], color='red', linewidth=1.0, label='Correct Trajectory')
            ax.scatter(xt_true[:, 0][0], xt_true[:, 1][0], xt_true[:, 2][0], color='green', label='Start')
            ax.scatter(xt_true[:, 0][-1], xt_true[:, 1][-1], xt_true[:, 2][-1], color='red', label='End')
            if title:
                ax.set_title(title)
            ax.legend()
            ax.grid(True) 
        
    else: # n_sample == 4 
        if dim_x == 1:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()
            for i in range(n_sample):
                axs[i].plot(time_vector, pred_0_list[:, i], color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
                axs[i].plot(time_vector, xt_true[:, i], color='red', linewidth=1.0, label='Correct Trajectory')
                axs[i].scatter(time_vector[0], xt_true[0, i], color='green', label='Start')
                axs[i].scatter(time_vector[-1], xt_true[-1, i], color='red', label='End')
                if title:
                    ax.set_title(title)
                axs[i].legend()
                axs[i].grid(True)
        elif dim_x == 2:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()
            for i in range(n_sample):
                axs[i].plot(pred_0_list[:, i, 0], pred_0_list[:, i, 1], color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
                axs[i].plot(xt_true[:, i, 0], xt_true[:, i, 1], color='red', linewidth=1.0, label='Correct Trajectory')
                axs[i].scatter(xt_true[0, i, 0], xt_true[0, i, 1], color='green', label='Start')
                axs[i].scatter(xt_true[-1, i, 0], xt_true[-1, i, 1], color='red', label='End')
                if title:
                    ax.set_title(title)
                axs[i].legend()
                axs[i].grid(True)
        else: # dim_x == 3
            fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': '3d'})
            axs = axs.flatten()
            for i in range(n_sample):
                axs[i].plot(pred_0_list[:, i, 0], pred_0_list[:, i, 1], pred_0_list[:, i, 2], color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
                axs[i].plot(xt_true[:, i, 0], xt_true[:, i, 1], xt_true[:, i, 2], color='red', linewidth=1.0, label='Correct Trajectory')
                axs[i].scatter(xt_true[0, i, 0], xt_true[0, i, 1], xt_true[0, i, 2], color='green', label='Start')
                axs[i].scatter(xt_true[-1, i, 0], xt_true[-1, i, 1], xt_true[-1, i, 2], color='red', label='End')
                if title:
                    ax.set_title(title)
                axs[i].legend()
                axs[i].grid(True)







########### same functions, designed to deal with dynamic time window ########### 


def evaluate_RMSE_d(model, ode, freq, n_sample, init_high, init_low, T0, T, dim_k=1):
    # function computing the RMSE (and the MSE) of a given model: 

    np.random.seed(666)
    dt = 1 / freq

    # true trajectories:
    dg_true = data.DataGenerator(ode, T, freq, n_sample, noise_sigma=0., init_high=init_high, init_low=init_low)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    if T0: # if T0>0, cut portion [0,T0] 
        xt_true = xt_true[T0*freq:, :, :]
    #print(np.shape(xt_true))


    # estimated trajectories:
    pred_0_list = []
    for i in range(n_sample):

        correct_param = xt_true[0, i, -dim_k:]
        #print(np.shape(correct_param))

        t = np.arange(T0,T,dt)
        T_plot = len(t)
        test_params = np.tile(correct_param, (T_plot,1))
        if dim_k != 0:
            pred_0 = model.simulate(xt_true[0, i, :][:-dim_k], t= t[:T_plot], u = test_params)
        else: # dim_k == 0
            pred_0 = model.simulate(xt_true[0, i, :], t= t[:T_plot])
        pred_0_list.append(pred_0)
    #print(np.shape(pred_0_list))


    xt_true = xt_true[:len(pred_0), :, :]
    if dim_k != 0:
        xt_true = xt_true[:, :, :-dim_k]
    xt_true = xt_true.squeeze()
    #print(np.shape(xt_true)) #(151, 25, 2)
    pred_0_list = np.transpose(pred_0_list, (1, 0, 2))
    pred_0_list = pred_0_list.squeeze()
    #print(np.shape(pred_0_list)) #(151, 25, 2)

    # RMSE:
    rmse_0_list = []
    mse_0_list = []
    for i in range(n_sample):
        rmse_0 = root_mean_squared_error(xt_true[:, i], pred_0_list[:,i]) 
        mse_0 = mean_squared_error(xt_true[:, i], pred_0_list[:,i])
        rmse_0_list.append(rmse_0)
        mse_0_list.append(mse_0)
    rmse_0 = np.mean(rmse_0_list)
    mse_0 = np.mean(mse_0_list)
    return rmse_0, mse_0




def evaluate_traj_d(model, ode, freq, n_sample, init_high, init_low,  T0, T, dim_x=1, dim_k=1, title=None):
    # function plotting the true and estimated trajectories:# function displaying the estimated trajectories:

    assert n_sample in [1, 4], "n_sample should be 1 or 4."

    np.random.seed(666)
    dt = 1 / freq
    time_vector = np.arange(T0, T + dt, dt)
    t = np.arange(T0, T, dt)
    T_plot = len(t)

    # true trajectories:
    dg_true = data.DataGenerator(ode, T, freq, n_sample, noise_sigma=0., init_high=init_high, init_low=init_low)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    if T0: # if T0>0, cut portion [0,T0] 
        xt_true = xt_true[T0*freq:, :, :]
    #print(np.shape(xt_true))


    # estimated trajectories:
    pred_0_list = []
    for i in range(n_sample):
        correct_param = xt_true[0, i, -dim_k:]
        #print(correct_param)
        t = np.arange(T0, T, dt)
        T_plot = len(t)
        test_params = np.tile(correct_param, (T_plot,1))
        if dim_k != 0:
            pred_0 = model.simulate(xt_true[0, i, :][:-dim_k], t= t[:T_plot], u = test_params)
        else:
            pred_0 = model.simulate(xt_true[0, i, :], t= t[:T_plot])
        pred_0_list.append(pred_0)
    #print(np.shape(pred_0_list))

    xt_true = xt_true[:len(pred_0), :, :]
    #print(np.shape(xt_true))
    time_vector = time_vector[:len(pred_0)]
    if dim_k != 0: # RMK!!! nel caso dim_k=0, xt_true = xt_true[:, :, :-dim_k] elimina la colonna -1, con output avente shape: (300, 1, 0) anziché (300, 1, 2)
        xt_true = xt_true[:, :, :-dim_k]
    #print(np.shape(xt_true))
    xt_true = xt_true.squeeze()
    #print(np.shape(xt_true))
    pred_0_list = np.transpose(pred_0_list, (1, 0, 2))
    pred_0_list = pred_0_list.squeeze()
    #print(np.shape(pred_0_list))

    if n_sample == 1:
        if dim_x == 1:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.plot(time_vector, pred_0_list, color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
            ax.plot(time_vector, xt_true, color='red', linewidth=1.0, label='Correct Trajectory')
            ax.scatter(time_vector[0], xt_true[0], color='green', label='Start')
            ax.scatter(time_vector[-1], xt_true[-1], color='red', label='End')
            if title:
                ax.set_title(title)
            ax.legend()
            ax.grid(True)
        elif dim_x == 2:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            ax.plot(pred_0_list[:, 0], pred_0_list[:, 1], color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
            ax.plot(xt_true[:, 0], xt_true[:, 1], color='red', linewidth=1.0, label='Correct Trajectory')
            ax.scatter(xt_true[:, 0][0], xt_true[:, 1][0], color='green', label='Start')
            ax.scatter(xt_true[:, 0][-1], xt_true[:, 1][-1], color='red', label='End')
            if title:
                ax.set_title(title)
            ax.legend()
            ax.grid(True) 
        else: # dim_x == 3
            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
            ax.plot(pred_0_list[:, 0], pred_0_list[:, 1], pred_0_list[:, 2], color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
            ax.plot(xt_true[:, 0], xt_true[:, 1], xt_true[:, 2], color='red', linewidth=1.0, label='Correct Trajectory')
            ax.scatter(xt_true[:, 0][0], xt_true[:, 1][0], xt_true[:, 2][0], color='green', label='Start')
            ax.scatter(xt_true[:, 0][-1], xt_true[:, 1][-1], xt_true[:, 2][-1], color='red', label='End')
            if title:
                ax.set_title(title)
            ax.legend()
            ax.grid(True) 
        
    else: # n_sample == 4 
        if dim_x == 1:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()
            for i in range(n_sample):
                axs[i].plot(time_vector, pred_0_list[:, i], color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
                axs[i].plot(time_vector, xt_true[:, i], color='red', linewidth=1.0, label='Correct Trajectory')
                axs[i].scatter(time_vector[0], xt_true[0, i], color='green', label='Start')
                axs[i].scatter(time_vector[-1], xt_true[-1, i], color='red', label='End')
                if title:
                    ax.set_title(title)
                axs[i].legend()
                axs[i].grid(True)
        elif dim_x == 2:
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()
            for i in range(n_sample):
                axs[i].plot(pred_0_list[:, i, 0], pred_0_list[:, i, 1], color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
                axs[i].plot(xt_true[:, i, 0], xt_true[:, i, 1], color='red', linewidth=1.0, label='Correct Trajectory')
                axs[i].scatter(xt_true[0, i, 0], xt_true[0, i, 1], color='green', label='Start')
                axs[i].scatter(xt_true[-1, i, 0], xt_true[-1, i, 1], color='red', label='End')
                if title:
                    ax.set_title(title)
                axs[i].legend()
                axs[i].grid(True)
        else: # dim_x == 3
            fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': '3d'})
            axs = axs.flatten()
            for i in range(n_sample):
                axs[i].plot(pred_0_list[:, i, 0], pred_0_list[:, i, 1], pred_0_list[:, i, 2], color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
                axs[i].plot(xt_true[:, i, 0], xt_true[:, i, 1], xt_true[:, i, 2], color='red', linewidth=1.0, label='Correct Trajectory')
                axs[i].scatter(xt_true[0, i, 0], xt_true[0, i, 1], xt_true[0, i, 2], color='green', label='Start')
                axs[i].scatter(xt_true[-1, i, 0], xt_true[-1, i, 1], xt_true[-1, i, 2], color='red', label='End')
                if title:
                    ax.set_title(title)
                axs[i].legend()
                axs[i].grid(True)




def evaluate_RMSE_d_1D(model, ode, freq, n_sample, init_high, init_low, T0, T, x_id=0, dim_k=1):
    # function computing the RMSE (and the MSE) of a given model, along a specific dimension of the state(x_id): 

    np.random.seed(666)
    dt = 1 / freq

    # true trajectories:
    dg_true = data.DataGenerator(ode, T, freq, n_sample, noise_sigma=0., init_high=init_high, init_low=init_low)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    if T0: # if T0>0, cut portion [0,T0] 
        xt_true = xt_true[T0*freq:, :, :]
    #print(np.shape(xt_true))


    # estimated trajectories:
    pred_0_list = []
    for i in range(n_sample):

        correct_param = xt_true[0, i, -dim_k:]
        #print(np.shape(correct_param))

        t = np.arange(T0,T,dt)
        T_plot = len(t)
        test_params = np.tile(correct_param, (T_plot,1))
        if dim_k != 0:
            pred_0 = model.simulate(xt_true[0, i, :][:-dim_k], t= t[:T_plot], u = test_params)
        else: # dim_k == 0
            pred_0 = model.simulate(xt_true[0, i, :], t= t[:T_plot])
        pred_0_list.append(pred_0)
    #print(np.shape(pred_0_list))


    xt_true = xt_true[:len(pred_0), :, :]
    if dim_k != 0:
        xt_true = xt_true[:, :, :-dim_k]
    xt_true = xt_true.squeeze()
    #print(np.shape(xt_true)) #(151, 25, 2)
    pred_0_list = np.transpose(pred_0_list, (1, 0, 2))
    pred_0_list = pred_0_list.squeeze()
    #print(np.shape(pred_0_list)) #(151, 25, 2)

    # get the specific dimension:
    xt_true = xt_true[:, :, x_id]
    #print(np.shape(xt_true))
    pred_0_list = pred_0_list[:, :, x_id]
    #print(np.shape(pred_0_list))

    # RMSE:
    rmse_0_list = []
    mse_0_list = []
    for i in range(n_sample):
        rmse_0 = root_mean_squared_error(xt_true[:, i], pred_0_list[:,i]) 
        mse_0 = mean_squared_error(xt_true[:, i], pred_0_list[:,i])
        rmse_0_list.append(rmse_0)
        mse_0_list.append(mse_0)
    rmse_0 = np.mean(rmse_0_list)
    mse_0 = np.mean(mse_0_list)
    return rmse_0, mse_0



def evaluate_traj_d_1D(model, ode, freq, n_sample, init_high, init_low,  T0, T, x_id, dim_x=1, dim_k=1, title=None, plot=True):
    # function plotting the true and estimated trajectories, along a specific dimension of the state(x_id): 

    assert n_sample in [1, 4], "n_sample should be 1 or 4."

    np.random.seed(666)
    dt = 1 / freq
    time_vector = np.arange(T0, T + dt, dt)
    t = np.arange(T0, T, dt)
    T_plot = len(t)

    # true trajectories:
    dg_true = data.DataGenerator(ode, T, freq, n_sample, noise_sigma=0., init_high=init_high, init_low=init_low)
    xt_true = dg_true.xt
    #print(np.shape(xt_true))

    if T0: # if T0>0, cut portion [0,T0] 
        xt_true = xt_true[T0*freq:, :, :]
    #print(np.shape(xt_true))


    # estimated trajectories:
    pred_0_list = []
    for i in range(n_sample):
        correct_param = xt_true[0, i, -dim_k:]
        #print(correct_param)
        t = np.arange(T0, T, dt)
        T_plot = len(t)
        test_params = np.tile(correct_param, (T_plot,1))
        if dim_k != 0:
            pred_0 = model.simulate(xt_true[0, i, :][:-dim_k], t= t[:T_plot], u = test_params)
        else:
            pred_0 = model.simulate(xt_true[0, i, :], t= t[:T_plot])
        pred_0_list.append(pred_0)
    #print(np.shape(pred_0_list))

    xt_true = xt_true[:len(pred_0), :, :]
    #print(np.shape(xt_true))
    time_vector = time_vector[:len(pred_0)]
    if dim_k != 0: # RMK!!! nel caso dim_k=0, xt_true = xt_true[:, :, :-dim_k] elimina la colonna -1, con output avente shape: (300, 1, 0) anziché (300, 1, 2)
        xt_true = xt_true[:, :, :-dim_k]
    #print(np.shape(xt_true))
    xt_true = xt_true.squeeze()
    #print(np.shape(xt_true)) # (150, 2)
    pred_0_list = np.transpose(pred_0_list, (1, 0, 2))
    pred_0_list = pred_0_list.squeeze()
    #print(np.shape(pred_0_list)) # (150, 2)

    # get the specific dimension:
    xt_true = xt_true[:, x_id]
    #print(np.shape(xt_true))
    pred_0_list = pred_0_list[:, x_id]
    #print(np.shape(pred_0_list))

    if plot:
        if n_sample == 1:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.plot(time_vector, pred_0_list, color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
            ax.plot(time_vector, xt_true, color='red', linewidth=1.0, label='Correct Trajectory')
            ax.scatter(time_vector[0], xt_true[0], color='green', label='Start')
            ax.scatter(time_vector[-1], xt_true[-1], color='red', label='End')
            if title:
                ax.set_title(title)
            ax.legend()
            ax.grid(True)

        else: # n_sample == 4 
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs = axs.flatten()
            for i in range(n_sample):
                axs[i].plot(time_vector, pred_0_list[:, i], color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
                axs[i].plot(time_vector, xt_true[:, i], color='red', linewidth=1.0, label='Correct Trajectory')
                axs[i].scatter(time_vector[0], xt_true[0, i], color='green', label='Start')
                axs[i].scatter(time_vector[-1], xt_true[-1, i], color='red', label='End')
                if title:
                    ax.set_title(title)
                axs[i].legend()
                axs[i].grid(True)
    
    return xt_true, pred_0_list

























########### same functions, designed to deal with 1 equation at a time: ###########

def SINDy_data_1D(ode_name, ode_param, freq, n_sample, noise_ratio, dim_x, dim_k, idx):

    np.random.seed(999)
    alg = 'tv'

    ode = equations.get_ode(ode_name, ode_param)
    T = ode.T
    init_low = ode.init_low 
    init_high = ode.init_high 
    has_coef = ode.has_coef 
    noise_sigma = ode.std_base * noise_ratio

    # data simulation: 
    dg = data.DataGenerator(ode, T, freq, n_sample, noise_sigma, init_low, init_high) 
    yt = dg.generate_data()
    #print(np.shape(yt)) 
    # print(yt[0:20, 0, 0]) 
    # print(yt[0:20, 0, 1]) 
    # print(yt[0:20, 0, 2])

    # numerical differentiation:
    if noise_sigma == 0:
        dxdt_hat = (yt[1:, :, :] - yt[:-1, :, :]) / (dg.solver.t[1:] - dg.solver.t[:-1])[:, None, None]
    else:
        dxdt_hat = num_diff(yt, dg, alg)
    #print(np.shape(dxdt_hat))
    #print('Numerical differentiation: Done.')


    # build dataset:
    X_train = yt[:-1, :, idx] # yt[:-1, :, :-1]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_train = np.transpose(X_train, (1, 0, 2))
    dX_train = dxdt_hat[:, :, idx] # dxdt_hat[:, :, :-1]
    dX_train = dX_train.reshape(dX_train.shape[0], dX_train.shape[1], 1)
    dX_train = np.transpose(dX_train, (1, 0, 2))

    if dim_k != 0 or dim_x > 1:
        param_train = np.delete(yt[:-1, :, :], idx, axis=2)
        #print(np.shape(param_train))
        param_train = np.transpose(param_train, (1, 0, 2))
        #print(np.shape(param_train))
        param_train = param_train.squeeze()
        #print(np.shape(param_train))

    X_list = []
    dX_list = []
    param_list = []
    n_train = len(X_train)
    for i in range(n_train):
        X_list.append(X_train[i])
        dX_list.append(dX_train[i])
        if dim_k != 0 or dim_x > 1:
            param_list.append(param_train[i])
    
    if dim_x == 1:
        feature_names = ["X0"]
        if dim_k == 1:
            feature_names += ["X1"]
        elif dim_k == 2:
            feature_names += ["X1", "X2"]
    elif dim_x == 2:
        if idx == 0:
            feature_names = ["X0", "X1"]
        else:
            feature_names = ["X1", "X0"]
        if dim_k == 1:
            feature_names += ["X2"]
    else: # dim_x == 3
        feature_names = ["X0", "X1", "X2"]
        if dim_k == 1:
            feature_names += ["X3"]

    return X_list, dX_list, param_list, feature_names



def evaluate_RMSE_1D(model, ode, freq, n_sample, init_high, init_low, dim_x, dim_k=1, idx=0):
    # function computing the RMSE (and the MSE) of a given model: 

    np.random.seed(666)
    dt = 1 / freq
    TIME = min(ode.T, 100) 

    # true trajectories:
    dg_true = data.DataGenerator(ode, TIME, freq, n_sample, noise_sigma=0., init_high=init_high, init_low=init_low)
    xt_true = dg_true.xt
    print(np.shape(xt_true))

    xt_true_1D = xt_true[:, :, idx]
    print(np.shape(xt_true_1D))

    xt_true_res = np.delete(xt_true, idx, axis=2)
    print(np.shape(xt_true_res))

    # estimated trajectories:
    pred_0_list = []
    for i in range(n_sample):

        correct_param = xt_true_res[:, i, :]
        print(np.shape(correct_param))
        
        t = np.arange(0,TIME,dt)
        T_plot = len(t)
        test_params = correct_param[:T_plot, :]
        print(np.shape(test_params))
        
        if dim_k != 0:
            x0 = np.array([xt_true_1D[0, i]]) if np.isscalar(xt_true_1D[0, i]) else xt_true_1D[0, i]
            print(np.shape(x0))
            pred_0 = model.simulate(x0, t= t[:T_plot], u = test_params)
        else: # dim_k == 0
            x0 = np.array([xt_true_1D[0, i]]) if np.isscalar(xt_true_1D[0, i]) else xt_true_1D[0, i]
            pred_0 = model.simulate(x0, t= t[:T_plot])
        pred_0_list.append(pred_0)
    print(np.shape(pred_0_list))


    xt_true_1D = xt_true_1D[:len(pred_0), :]
    print(np.shape(xt_true_1D)) #(151, 25, 2)
    pred_0_list = np.transpose(pred_0_list, (1, 0, 2))
    pred_0_list = pred_0_list.squeeze()
    print(np.shape(pred_0_list)) #(151, 25, 2)

    # RMSE:
    rmse_0_list = []
    mse_0_list = []
    for i in range(n_sample):
        rmse_0 = root_mean_squared_error(xt_true_1D[:, i], pred_0_list[:,i]) 
        mse_0 = mean_squared_error(xt_true_1D[:, i], pred_0_list[:,i])
        rmse_0_list.append(rmse_0)
        mse_0_list.append(mse_0)
    rmse_0 = np.mean(rmse_0_list)
    mse_0 = np.mean(mse_0_list)
    return rmse_0, mse_0





def evaluate_RMSE_ciao(model, ode, freq, n_sample, init_high, init_low, dim_x, dim_k=1, mono_dim = False, idx=0):
    # function computing the RMSE of a given model: 

    np.random.seed(666)
    dt = 1 / freq

    TIME = min(ode.T, 10)
    

    # true trajectories:
    dg_true = data.DataGenerator(ode, TIME, freq, n_sample, noise_sigma=0., init_high=init_high, init_low=init_low)
    xt_true = dg_true.xt
    print(np.shape(xt_true))


    if mono_dim and dim_x == 2: # non funzionerà nel caso dim_x>2 -> todo
        xt_true_copy = xt_true
        idx_del = dim_x - 1 - idx
        xt_true = np.delete(xt_true, idx_del, axis=2)
        print(np.shape(xt_true))
    # dim_x == 1 -> ok
    # dim_x == 2 -> idx_del = dim_x - 1 - idx


    # estimated trajectories:
    pred_0_list = []
    for i in range(n_sample):

        if mono_dim:
            xt_true_copy = np.delete(xt_true_copy, idx, axis=2)
            correct_param = xt_true_copy[0, i, :]
        else:
            correct_param = xt_true[0, i, -dim_k:]
        print(np.shape(correct_param))

        t = np.arange(0,TIME,dt)
        T_plot = len(t)
        test_params = np.tile(correct_param, (T_plot,1))
        if dim_k != 0:
            pred_0 = model.simulate(xt_true[0, i, :][:-dim_k], t= t[:T_plot], u = test_params)
        else: # dim_k == 0
            if mono_dim and dim_x > 1:
                pred_0 = model.simulate(xt_true[0, i, :], t= t[:T_plot], u = test_params)
            else:
                pred_0 = model.simulate(xt_true[0, i, :], t= t[:T_plot])
        pred_0_list.append(pred_0)
    print(np.shape(pred_0_list))


    
    xt_true = xt_true[:len(pred_0), :, :]
    if dim_k != 0:
        xt_true = xt_true[:, :, :-dim_k]
    xt_true = xt_true.squeeze()
    print(np.shape(xt_true)) #(151, 25, 2)
    pred_0_list = np.transpose(pred_0_list, (1, 0, 2))
    pred_0_list = pred_0_list.squeeze()
    print(np.shape(pred_0_list)) #(151, 25, 2)

    # RMSE:
    rmse_0_list = []
    for i in range(n_sample):
        rmse_0 = mean_squared_error(xt_true[:, i], pred_0_list[:,i]) 
        rmse_0_list.append(rmse_0)
    rmse_0 = np.mean(rmse_0_list)
    return rmse_0



def evaluate_MSE_1D_ciao(model, ode, freq, n_sample, init_high, init_low, dim_x, dim_k=1, idx=0):
    # function computing the RMSE of a given model: 

    np.random.seed(666)
    dt = 1 / freq

    TIME = min(ode.T, 10)

    # true trajectories:
    dg_true = data.DataGenerator(ode, TIME, freq, n_sample, noise_sigma=0., init_high=init_high, init_low=init_low)
    xt_true = dg_true.xt
    print(np.shape(xt_true))

    # estimated trajectories:
    pred_0_list = []
    for i in range(n_sample):
        #correct_param = xt_true[0, i, -dim_k:]
        correct_param = np.delete(xt_true[0, i, :], idx, axis=2)
        #print(correct_param)
        t = np.arange(0,TIME,dt)
        T_plot = len(t)
        test_params = np.tile(correct_param, (T_plot,1))
        if dim_k != 0:
            #print(np.shape(xt_true[0, i, :][:-dim_k]))
            pred_0 = model.simulate(xt_true[0, i, :][:-dim_k], t= t[:T_plot], u = test_params)
        else: # dim_k == 0
            if dim_x == 1:
                pred_0 = model.simulate(xt_true[0, i, idx], t= t[:T_plot])
            else: 
                pred_0 = model.simulate(xt_true[0, i, idx], t= t[:T_plot], u = test_params)
        pred_0_list.append(pred_0)
    #print(np.shape(pred_0_list))


    
    xt_true = xt_true[:len(pred_0), :, :]
    if dim_k != 0:
        xt_true = xt_true[:, :, :-dim_k]
    xt_true = xt_true.squeeze()
    print(np.shape(xt_true)) #(151, 25, 2)
    pred_0_list = np.transpose(pred_0_list, (1, 0, 2))
    pred_0_list = pred_0_list.squeeze()
    print(np.shape(pred_0_list)) #(151, 25, 2)

    # RMSE:
    rmse_0_list = []
    for i in range(n_sample):
        rmse_0 = mean_squared_error(xt_true[:, i], pred_0_list[:,i]) 
        rmse_0_list.append(rmse_0)
    rmse_0 = np.mean(rmse_0_list)
    return rmse_0

  