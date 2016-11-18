from scipy import optimize


class trainer(object):
    
    def __init__(self, neural_network):
        self.neural_network = neural_network
        
    def callback(self, params):
        self.neural_network.set_parameters(params) 
    
    #Funcion que envuelve a la funcion de costo para que el metodo minimize funcione
    def cost_function_wrapper(self, params, x, y):
        self.neural_network.set_parameters(params)
        cost = self.neural_network.cost_function(x, y)
        grad = self.neural_network.gradients(x,y)
        return cost, grad
        
    def train(self, x, y):
        #Make an internal variable for the callback function:
        self.x = x
        self.y = y
        
        params = self.neural_network.get_parameters()
        
        options = {'maxiter': 300, 'disp' : False}
        min_value = optimize.minimize(self.cost_function_wrapper, params, jac=True, method='BFGS', 
                                 args=(x, y), options=options, callback=self.callback)

        self.neural_network.set_parameters(min_value.x)
        self.optimizationResults = min_value