import numpy as np
import sympy as sp
import pysindy as ps
import dill
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from D_CODE.run_simulation import run as run_SRT
from D_CODE.run_simulation_vi import run as run_DCODE
from D_CODE.run_simulation_vi import run_HD as run_DCODE_HD
from data import SINDy_data


def set_param_freq(ode_param, freq):
    if ode_param is not None:
        param = [float(x) for x in ode_param.split(',')]
    else:
        param = None
    if freq >= 1:
        freq = int(freq)
    else:
        freq = freq
    return param, freq

def SRT_simulation(ode_name, param, x_id, freq, n_sample, noise_sigma, alg, seed, n_seed, idx=0, T0=0, T=15):
    print("Running with: ode_name={}, ode_param={}, x_id={}, freq={}, n_sample={}, noise_sigma={}, alg={}, seed={}, n_seed={}".format(
    ode_name, param, x_id, freq, n_sample, noise_sigma, alg, seed, n_seed))
    building_blocks_lambda, function_names = run_SRT(ode_name, param, x_id, freq, n_sample, noise_sigma, alg, seed=seed, n_seed=n_seed, idx=idx, T0=T0, T=T)
    return building_blocks_lambda, function_names


def D_CODE_simulation(ode_name, param, x_id, freq, n_sample, noise_sigma, seed, n_seed, T0=0, T=15):
    print("Running with: ode_name={}, ode_param={}, x_id={}, freq={}, n_sample={}, noise_sigma={}, seed={}, n_seed={}".format(
    ode_name, param, x_id, freq, n_sample, noise_sigma, seed, n_seed))
    building_blocks_lambda, function_names = run_DCODE(ode_name, param, x_id, freq, n_sample, noise_sigma, seed=seed, n_seed=n_seed, T0=T0, T=T)
    return building_blocks_lambda, function_names

def D_CODE_simulation_HD(ode_name, param, x_id, freq, n_sample, noise_sigma, seed, n_seed, T0=0, T=15, latent_data=None):
    print("Running with: ode_name={}, ode_param={}, x_id={}, freq={}, n_sample={}, noise_sigma={}, seed={}, n_seed={}".format(
    ode_name, param, x_id, freq, n_sample, noise_sigma, seed, n_seed))
    building_blocks_lambda, function_names = run_DCODE_HD(ode_name, param, x_id, freq, n_sample, noise_sigma, seed=seed, n_seed=n_seed, T0=T0, T=T, latent_data=latent_data)
    return building_blocks_lambda, function_names



def intercept_library_fun(n_variables):
    
    if n_variables == 1:
        X0 = sp.symbols('X0')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0), intercept, modules='numpy')
        function_name_intercept = lambda X0: str(intercept)    
    elif n_variables == 2:
        X0, X1 = sp.symbols('X0 X1')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0, X1), intercept, modules='numpy')
        function_name_intercept = lambda X0, X1: str(intercept)
    elif n_variables == 3:
        X0, X1, X2 = sp.symbols('X0 X1 X2')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0, X1, X2), intercept, modules='numpy')
        function_name_intercept = lambda X0, X1, X2: str(intercept)        
    elif n_variables == 4:
        X0, X1, X2, X3 = sp.symbols('X0 X1 X2 X3')
        intercept = sp.Integer(1)
        intercept_lambda = sp.lambdify((X0, X1, X2, X3), intercept, modules='numpy')
        function_name_intercept = lambda X0, X1, X2, X3: str(intercept)

    intercept_library = ps.CustomLibrary(library_functions=[intercept_lambda], function_names=[function_name_intercept])
    return intercept_library



def bb_combinations(building_blocks_lambda_0, building_blocks_lambda_1, function_names_0, function_names_1, init_high, init_low, dim_x, dim_k):


    # remove repeated building blocks:
    N = 10
    tol = 1e-3

    if dim_x == 1:
        x_samples = np.random.uniform(init_low, init_high, N).reshape(-1, 1) 
    else:
        x_samples = np.empty((dim_x+dim_k, N))
        for i in range(dim_x+dim_k):
            x_samples[i, :] = np.random.uniform(init_low[i], init_high[i], N) 
    x_samples = np.array(x_samples)
    #print(np.shape(x_samples))

    f_samples = []
    for i in range(len(building_blocks_lambda_0)):
        f_hat = building_blocks_lambda_0[i]
        #aux = [f_hat(x_samples[0, i], x_samples[1, i], x_samples[2, i]) for i in range(x_samples.shape[1])]
        aux = [f_hat(*x_samples[:, i]) for i in range(x_samples.shape[1])]
        f_samples.append(aux)
    #print(np.shape(f_samples))

    building_blocks_lambda_1_fil = [] # filter building blocks from eq. 2
    function_names_1_fil = []
    for i in range(len(building_blocks_lambda_1)):

        flag = 1
        f_hat = building_blocks_lambda_1[i] 
        #aux = [f_hat(x_samples[0, i], x_samples[1, i], x_samples[2, i]) for i in range(x_samples.shape[1])]
        aux = [f_hat(*x_samples[:, i]) for i in range(x_samples.shape[1])]
        for j in range(len(f_samples)):
            if root_mean_squared_error(aux, f_samples[j]) < tol: # filter similar subprograms
                flag = 0
        if flag:
            f_samples.append(aux)
            building_blocks_lambda_1_fil.append(building_blocks_lambda_1[i])
            function_names_1_fil.append(function_names_1[i])


    # combine building blocks from eq. 1 and eq. 2:
    combined_bb = [
    [lambda_0, lambda_1] 
    for lambda_0 in building_blocks_lambda_0 
    for lambda_1 in building_blocks_lambda_1_fil
    ]
    combined_fn = [
        [fn_0, fn_1] 
        for fn_0 in function_names_0 
        for fn_1 in function_names_1_fil
    ]
    bb_0 = [
        [lambda_0] 
        for lambda_0 in building_blocks_lambda_0 
    ]
    fn_0 = [
        [fn_0] 
        for fn_0 in function_names_0 
    ]
    bb_1 = [
        [lambda_1] 
        for lambda_1 in building_blocks_lambda_1_fil
    ]
    fn_1 = [
        [fn_1] 
        for fn_1 in function_names_1_fil
    ]

    bbs = bb_0 + bb_1 + combined_bb
    fns = fn_0 + fn_1 + combined_fn

    return bbs, fns




def smart_SINDy_model(ode_name, ode_param, x_id, freq_SR, n_sample, noise_ratio, alg, seed, n_seed, T0, T, X_list_t, dX_list_t, param_list, feature_names, dim_x, dim_k, freq, dt, ode, building_blocks_lambda, function_names, lazy, patience):

    if building_blocks_lambda is None or patience > 4: # either SR has never been called or patience is over (the found building blocks have not been useful over last 5 iterations)
        print('')
        print('Searching for additonal building blocks -> SR-T call:')
        print('')

        if lazy == False:
            # SR-T call:
            building_blocks_lambda, function_names = SRT_simulation(ode_name, ode_param, x_id, freq_SR, n_sample, noise_ratio, alg, seed=seed, n_seed=n_seed, T0=T0, T=T)

            # save building blocks: 
            with open('oscillating_selkov_bb.pkl', 'wb') as f:
                dill.dump((building_blocks_lambda, function_names), f)
        else:
            # load building blocks:
            with open('oscillating_selkov_bb.pkl', 'rb') as f:
                building_blocks_lambda, function_names = dill.load(f)
        patience = 0
    

    # print('')
    # print('Searching for the best building block:')
    patience += 1
    errors = []
    n_features_vec = []
    intercept_library = intercept_library_fun(dim_x+dim_k) # intercept library
    polynomial_library = ps.PolynomialLibrary(degree=3, include_bias=False) # polynomial library

    for i in range(len(building_blocks_lambda)):

        # building the library:
        custom_library = ps.CustomLibrary(library_functions=[building_blocks_lambda[i]], function_names=[function_names[i]]) # custom library with building block
        generalized_library = ps.ConcatLibrary([polynomial_library, custom_library]) # enlarged library, adding the building block to polynomial library
        final_library = ps.ConcatLibrary([intercept_library, generalized_library]) # add the intercept

        # fitting the model:
        model = ps.SINDy(feature_names=feature_names, feature_library=final_library, optimizer=ps.STLSQ(threshold=0.09))
        model.fit(X_list_t, t=dt, multiple_trajectories=True, x_dot=dX_list_t)
        # print('Model:')
        # model.print()   

        # print('')
        # print('library:')
        # library_terms = final_library.get_feature_names(input_features=feature_names)
        # for term in library_terms:
        #     print(term)    

        # evaluate the model:  
        coefficients = model.coefficients()
        model_complexity = np.count_nonzero(np.array(model.coefficients()))
        lasso_penalty = np.sum(np.abs(coefficients))
        if model_complexity < 20 and lasso_penalty < 20: #filter too complex models (for sure not correct and likely to crash the code):
            _, mse = SINDy_data.evaluate_RMSE_d(model, ode, freq, 10, ode.init_high, ode.init_low, T0, T, dim_k) # compute MSE      
            alpha = 0.01 # regularization parameter
            error = mse + alpha * lasso_penalty # final evaluation metric
            #print('error:', error)
        else:
            error = 1000
            #print('Too complex model')
        #print('')
        errors.append(error)
        n_features_vec.append(np.count_nonzero(np.array(model.coefficients())))


    if all(err == 1000 for err in errors):
        print('No model update, all smart-SINDy models are too complex')
        return None, building_blocks_lambda, function_names, None, None, patience
    else:
        # Final model:
        min_error = min(errors)
        idxs = [i for i, e in enumerate(errors) if abs(e - min_error) < 0.01]
        n_features_vec_2 = [n_features_vec[i] for i in idxs]

        if len(idxs) > 1:
            # print('Multiple models with similar error, choosing the simplest one')
            # print('')
            idx = idxs[np.argmin(n_features_vec_2)]
        else:
            idx = idxs[0]

        # building the library:
        custom_library = ps.CustomLibrary(library_functions=[building_blocks_lambda[idx]], function_names=[function_names[idx]])  # custom library with building block
        model = ps.SINDy(feature_names=feature_names, feature_library=custom_library, optimizer=ps.STLSQ(threshold=0.01))
        model.fit(X_list_t, t=dt, multiple_trajectories=True, x_dot=dX_list_t)
        building_block = custom_library.get_feature_names(input_features=feature_names) 
        generalized_library = ps.ConcatLibrary([polynomial_library, custom_library]) # enlarged library, adding the building block to polynomial library
        final_library = ps.ConcatLibrary([intercept_library, generalized_library]) # add the intercept

        # fitting the model:
        model = ps.SINDy(feature_names=feature_names, feature_library=final_library, optimizer=ps.STLSQ(threshold=0.09))
        model.fit(X_list_t, t=dt, multiple_trajectories=True, x_dot=dX_list_t)


        # best building block:
        print('')
        print('Best building block:')
        print(building_block)
        print('')

        # final model:
        print('smart-SINDy model:')
        model.print()

        # save model and bb: 
        with open('oscillating_selkov_model.pkl', 'wb') as f:
            dill.dump((model, building_block), f)

        coefficients = model.coefficients()
        model_complexity = np.count_nonzero(np.array(model.coefficients()))
        print('Model complexity: ', model_complexity)
        lasso_penalty = np.sum(np.abs(coefficients))
        print('Lasso penalty: ', lasso_penalty)

        return model, building_blocks_lambda, function_names, model_complexity, lasso_penalty, patience
    



def SINDy_call(X_list, dX_list, param_list, feature_names, degree, include_bias, threshold, dt):

    print('SINDy model:')
    polynomial_library = ps.PolynomialLibrary(degree = degree, include_bias=include_bias)
    model = ps.SINDy(feature_names = feature_names, feature_library = polynomial_library, optimizer=ps.STLSQ(threshold=threshold))
    model.fit(X_list, t=dt, multiple_trajectories=True, x_dot = dX_list)
    model.print()

    coefficients = model.coefficients()
    model_complexity = np.count_nonzero(np.array(model.coefficients()))
    print('Model complexity: ', model_complexity)
    lasso_penalty = np.sum(np.abs(coefficients))
    print('Lasso penalty: ', lasso_penalty)
    
    return model, model_complexity, lasso_penalty





# Functions to plot experiment results:
# from Data.SINDy_data import evaluate_RMSE_d, evaluate_traj_d_1D
def plot(exp, x_id=0):
        plot_times_1 = exp.turning_points.copy() 
        plot_models_1 = exp.model_history.copy()
        plot_times_1.append(exp.H)
        plot_models_1.insert(0, plot_models_1[0])

        xt_true = []
        pred_list = []
        time_vector_1 = np.arange(0, plot_times_1[-1], exp.dt)
        time_vector_2 = np.arange(plot_times_1[1], plot_times_1[-1], exp.dt)
        for i in range(len(plot_models_1)):
            xt_true_i, pred_i = evaluate_traj_d_1D(plot_models_1[i], exp.ode, 10, 1, [0.05, 0.05, 0], [0.05, 0.05, 0], plot_times_1[i], plot_times_1[i+1], x_id, exp.dim_x, exp.dim_k, plot=False)
            if i == 0:
                xt_true = np.concatenate((xt_true, xt_true_i), axis = 0)
            else: 
                xt_true = np.concatenate((xt_true, xt_true_i), axis = 0)
                pred_list = np.concatenate((pred_list, pred_i), axis = 0)
                
            
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(time_vector_2, pred_list, color='blue', linewidth=1.0, alpha=0.7, label='Estimated Trajectory')
        ax.plot(time_vector_1, xt_true, color='red', linewidth=1.0, label='Correct Trajectory')
        ax.scatter(time_vector_1[0], xt_true[0], color='green', label='Start')
        ax.scatter(time_vector_1[-1], xt_true[-1], color='red', label='End')
        for x in [plot_times_1[1], plot_times_1[2]]: 
            ax.axvline(x=x, color='black', linestyle='--', linewidth=1.0, label='Vertical Line' if x == 1 else "")

        ax.set_xlabel('t')
        ax.set_ylabel('$x_{}(t)$'.format(x_id+1))
        ax.set_xlim(0.-3, exp.H+3)
        ax.legend()
        ax.set_title('Symbolic SINDy Model Discovery')
        ax.grid(True)

def plot_RMSE(exp):
        time_vector = np.arange(exp.turning_points[1], exp.H+1)
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(time_vector, exp.rmse_history, 'ro')
        ax.plot(time_vector, exp.rmse_history, 'r-', linewidth=0.3)
        for x in [exp.turning_points[1], exp.turning_points[2]]:  
            ax.axvline(x=x, color='black', linestyle='--', linewidth=1.0, label='Vertical Line' if x == 1 else "")
        ax.set_xlabel('t')
        ax.set_ylabel('RMSE')
        ax.set_xlim(0.-3, exp.H+3)
        ax.set_title('RMSE time-series')
