import numpy as np 
from scipy import interpolate
import matplotlib.pyplot as plt

class PPRterm : 
    
    def __init__(self, function, w) -> None:
        """Initialize the regressor term

        Args:
            function (func): initialization non linear function
            w (ndarray): initialization weights vector
        """
        self.function = function 
        self.w = w 
        
    def output(self, X) :
        """computes the output vector using the actual parameters

        Args:
            X (ndarray): input data matrix

        Returns:
            ndarray: the output vector 
        """
        return self.function(X@self.w) 
    
    def derivative(self, X) : 
        """computes the output gradient vector using the actual parameters

        Args:
            X (ndarray): input data matrix

        Returns:
            ndarray: the gradient vector
        """
        dv = 0.00000001 
        V = X@self.w 
        Vprev = V - dv
        
        Y = self.function(V)
        Y_prev = self.function(Vprev)
        
        return (Y-Y_prev)/dv
    
    def modify(self, g, w) :
        """Updates the parameters of the PPR term

        Args:
            g (func): the new non linear function
            w (ndarray): the new weights vector
        """
        self.function = g 
        self.w = w
        

class ProjectionPursuitRegressor() : 
    
    def __init__(self, function_init, w_init) -> None:
        """Initialize the regressor's parameters

        Args:
            function_init (func): initialization non linear function
            w_init (ndarray): initialization weights vector
        """
        self.function_init = function_init 
        self.w_init = w_init
        ppr_first = PPRterm(function_init, w_init)
        self.terms = {1 : ppr_first}
        
    def estimate_g(self, X, y, term_id, knot_numbers) : 
        """For a given projection vector w, this function computes the optimal function g using splines interpolation

        Args:
            X (ndarray): input data matrix
            y (ndarray): output data matrix
            term_id (int): the id of the PPR term
            knot_numbers (int): the number of knots for the splines smoother

        Returns:
            func: the optimal function g
        """
        
        v = X@self.terms[term_id].w
        
        # if term_id>1 :
        #     y_out = np.zeros(X.shape[0])
        #     for i in range(1, term_id) : 
        #         y_out += self.terms[i].output(X)  
                
        #     y -= y_out
        
        try : 
            A = np.argsort(v) 
            v, y = v[A], y[A]
            # plt.scatter(list(range(len(v))), v)
            # plt.show()
            # knot_numbers = 10
            x_new = np.linspace(0, 1, knot_numbers+2)[1:-1]
            q_knots = np.quantile(v, x_new)
            t,c,k = interpolate.splrep(v, y, t=q_knots, s=1)
            return interpolate.BSpline(t,c,k)
        except : 
            A = np.argsort(v) 
            v, y = v[A], y[A]
            plt.scatter(list(range(len(v))), v)
            plt.show()
        
    def estimate_w(self, X, y, term_id) : 
        """Computes the optimal weights vector by assimilating the optimization problem to a weighted least squares regression

        Args:
            X (ndarray): input data matrix
            y (ndarray): output data matrix
            term_id (int): PPR term id

        Returns:
            ndarray: optimal w
        """
        
        g_prime = self.terms[term_id].derivative(X)
        y_out = self.terms[term_id].output(X) 
        # y_out = self.predict(X)
        
        W = np.diag(g_prime**2)
        targets = X@self.terms[term_id].w + (y-y_out)/g_prime
        
        
        A = np.linalg.inv(X.T @ W @ X)
        
        B = X.T @ W 
        
        params = A@B@targets  
        
        return params
    
    def fit(self, X, y_init, Nb_terms, max_iteration, k_number, val_set=None) : 
        """Fits the regressor to the input/output vectors given 

        Args:
            X (ndarray): input data matrix
            y_init (ndarray): output data matrix
            Nb_terms (int): number of PPR terms
            max_iteration (int): maximum number of iterations to fit each PPR term
            k_number (int): number of knots to use for the splines smoothing

        Returns:
            ndarray: the loss evolution vector
        """
        
        y_current = np.copy(y_init)
        Error = []
        val_error = []
        for i in range(1, Nb_terms) : 
            if i not in self.terms : 
                w_init = np.random.normal(0, 1, X.shape[1])
                w_init /= np.linalg.norm(w_init)
                self.terms[i] = PPRterm(self.function_init, w_init) 
            for _ in range(max_iteration) : 
                # print(j)
                self.terms[i].function = self.estimate_g(X, y_current, i, knot_numbers=k_number) 
                w = self.estimate_w(X, y_current, i)  
                self.terms[i].w = w/np.linalg.norm(w)
                Error.append(self.loss(X, y_init))
                if val_set : 
                    val_error.append(self.loss(val_set[0], val_set[1]))
            y_current -= self.terms[i].output(X)
                
        return Error, val_error
    
    def predict(self, X) : 
        """Predicts the output given an input matrix

        Args:
            X (ndarray): input data matrix

        Returns:
            ndarray: predicted output data matrix
        """
        
        y = np.zeros(X.shape[0])
        for i in self.terms : 
            y += self.terms[i].output(X) 
            
        return y 
    
    def loss(self, X, y) : 
        """Computes the loss of the predictions with respect to the ground truth

        Args:
            X (ndarray): input data matrix
            y (ndarray): output data matrix

        Returns:
            float: squared error loss 
        """
        
        y_out = self.predict(X) 
        
        return (np.sum((y-y_out)**2))**0.5