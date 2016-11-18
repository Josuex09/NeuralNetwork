import numpy as np

class Neural_Network(object):
    
    def __init__(self, input_layer,hidden_layer,output_layer):        
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        
        np.random.seed(853)
        #Inicialmente los pesos de la capa oculta son escogidos aleatoriamente
        #w1 es una matriz del tamano de entradas por el tamano de la capa oculta
        #w2 es una matriz del tamano de la capa oculta por el tamano de la salida
        self.w1 = np.random.randn(self.input_layer,self.hidden_layer)
        self.w2 = np.random.randn(self.hidden_layer,self.output_layer)
        
    def forward_propagation(self, x):
        #Multiplicar las entradas por los valores de la primera sinapsis 
        self.z2 = np.dot(x, self.w1)
        #El valor de aplicar la funcion de activacion a cada elemento de la matriz actual
        self.a2 = self.activation_function(self.z2)
        #Multiplicar las entradas por los valores de la segunda sinapsis
        self.z3 = np.dot(self.a2, self.w2)
        #El valor de aplicar la funcion de activacion a cada elemento de la matriz final
        #Se obtiene la prediccion
        goal = self.activation_function(self.z3) 
        return goal
             
        
    def activation_function(self, z):
        return 1/(1+np.exp(-z))
    
    def activation_function_d(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    #Funcion que da el costo de correr un conjunto de datos
    def cost_function(self, x, y):
        goal = self.forward_propagation(x)
        j = 0.5*sum((y-goal)**2)
        return j
    
        
        
    def backpropagation(self, x, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        goal = self.forward_propagation(x)
        
        delta3 = np.multiply(-(y-goal), self.activation_function_d(self.z3))
        #Add gradient of regularization term:
        djdw2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.w2.T)*self.activation_function_d(self.z2)
        #Add gradient of regularization term:
        djdw1 = np.dot(x.T, delta2)
        
        return djdw1, djdw2
    
    
    def gradients(self, x, y):
        #Calcular la gradiente de w1 y w2
        djdw1, djdw2 = self.backpropagation(x, y)
        return np.concatenate((djdw1.ravel(), djdw2.ravel()))
    
    def get_parameters(self):
        parameters = np.concatenate((self.w1.ravel(), self.w2.ravel()))
        return parameters
    
    def set_parameters(self, params):
        w1_start = 0
        w1_end = self.hidden_layer * self.input_layer
        self.w1 = np.reshape(params[w1_start:w1_end], (self.input_layer , self.hidden_layer))
        w2_end = w1_end + self.hidden_layer*self.output_layer
        self.w2 = np.reshape(params[w1_end:w2_end], (self.hidden_layer, self.output_layer))