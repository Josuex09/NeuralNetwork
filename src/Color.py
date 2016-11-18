from PIL import Image
from sklearn import neural_network
from numpy import average
from audioop import reverse
import math

def round_closest(x):
    return int(round(x,0))


#Metodo que da la lista con la cantidad de colores que presenta una imagen
def get_colors_count(neural_network, image_path,class_array):  
    im = Image.open(image_path)
    rgb_im = im.convert('RGB')
    width, height = im.size
    color_counts = {}
  
    #Recorrer toda la imagen para evaluar cada pixel
    for i in range(0,width):
        for j in range(0,height):
            r,g,b = rgb_im.getpixel((i,j))
            #Normalizar los valores
            r = r/255.
            g = g/255.
            b = b/255.
            index = round_closest(neural_network.forward_propagation([r,g,b])*9)
            color = class_array[index]
            if color in color_counts:
                color_counts[color]+=1
            else:
                color_counts[color]= 1    
    return color_counts


def average(color_counts):
    acum = 0
    for color in color_counts:
        acum += color_counts[color]
    
    return float(acum) / float(len(color_counts))
        
#Metodo que calcula la desviacion estandar de los colores
def standard_deviation(color_counts):
    average_value = average(color_counts)
    acum = 0
    for color in color_counts:
        acum += (float(color_counts[color]) - average_value)**2
    return math.sqrt(float(acum)/float(len(color_counts)))
        
def get_predominant_color(neural_network, image_path, class_array):
    colors_count  = get_colors_count(neural_network, image_path, class_array)
    standard_deviation_value = standard_deviation(colors_count)
    sorted_colors = sorted(colors_count.items(), key=lambda(k,v):(v,k),reverse=True)
    if(len(sorted_colors) == 1):
        return sorted_colors
    else:
        predominant_colors = []
        max = 3
        #Agregar el primer color
        predominant_colors.append(sorted_colors[0][0])
        
        index = 1
        flag  = True
        while index<max and flag:
            if((sorted_colors[index-1][1] - sorted_colors[index][1]) <= standard_deviation_value):
                predominant_colors.append(sorted_colors[index][0])        
            else:
                flag = False
            index+=1
        
        return predominant_colors
    
    
    

    
    
    
    
    