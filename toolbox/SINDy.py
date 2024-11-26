import numpy as np
import pysindy as ps

class SINDy():

    def __init__(self, degree=3, include_bias=True, threshold=0.08):
        self.degree = degree
        self.include_bias = include_bias
        self.threshold = threshold


    def call(self, X_list, dX_list, param_list, feature_names, dt):

        
        polynomial_library = ps.PolynomialLibrary(degree=self.degree, include_bias=self.include_bias)
        model = ps.SINDy(feature_names=feature_names, feature_library=polynomial_library, optimizer=ps.STLSQ(threshold=self.threshold))
        model.fit(X_list, t=dt, multiple_trajectories=True, x_dot = dX_list)
        
        coefficients = model.coefficients()
        model_complexity = np.count_nonzero(np.array(model.coefficients()))
        
        lasso_penalty = np.sum(np.abs(coefficients))

        if model_complexity < 20 and lasso_penalty < 20:
            print('SINDy model:')
            model.print()
            print('Model complexity: ', model_complexity)
            print('Lasso penalty: ', lasso_penalty)
            return model, model_complexity, lasso_penalty
        else:
            print('Too complex')
            return None, model_complexity, lasso_penalty
