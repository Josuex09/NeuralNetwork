from NeuralNetwork import Neural_Network
from Trainer import trainer
from Color import  get_predominant_color
import numpy as np
from numpy import float128, dtype
from IPython.utils._tokenize_py2 import String
from Crypto.Util.number import size


data_filepath = "../colores.csv"
img_filepath = "../images/tanzania.jpg"

x = np.genfromtxt(data_filepath, dtype = np.float128,delimiter = ",", usecols=(0,1,2))
y = np.genfromtxt(data_filepath,dtype = np.string0, delimiter = ",", usecols=(3))

class_array = []
new_y = []

for i in range(y.size):
    if y[i] not in class_array:
        class_array.append(y[i])
    new_y.append([class_array.index(y[i])])
    
color_length = len(class_array)
new_y = np.array(new_y,dtype=float128)

x = x/255.
new_y = new_y/float(color_length-1)

NN = Neural_Network(3,11,1)

trainer = trainer(NN)
trainer.train(x, new_y)

print(get_predominant_color(NN,img_filepath,class_array))

