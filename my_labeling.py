__authors__ = ['1604284']
__group__ = 'GrupDL.17'

import collections

import numpy as np
import Kmeans
from Kmeans import *
import KNN
from KNN import *
from utils_data import read_dataset, visualize_k_means, visualize_retrieval
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))




def Retrieval_by_color(list_imagenes,etiquetes_Kmeans,pregunta ,porcentajes=None):

    arrayL = []
    for i in range(len(list_imagenes)):
        if pregunta in etiquetes_Kmeans[i]: #filtra las imagnes que no tiene la
            for etiqueta in range(len(etiquetes_Kmeans[i])):
                if pregunta == etiquetes_Kmeans[i][etiqueta]:
                    if porcentajes is not None:
                        arrayL.append([list_imagenes[i],porcentajes[i][etiqueta]])
                        break
                    else:
                        arrayL.append(list_imagenes[i])
                        break

    array=np.array(arrayL)
    if porcentajes is not None:
        array=array[array[:, 1].argsort(axis=1)]

    return array



def Retrieval_by_shape(list_imagenes,etiquetes_Knn,pregunta,porcentajes=None):
    arrayL = []

    for i in range(len(list_imagenes)):
        if pregunta == etiquetes_Knn[i]:
            if porcentajes is not None:
                arrayL.append([list_imagenes[i], porcentajes[i]])
            else:
                arrayL.append(list_imagenes[i])

    array=np.array(arrayL)
    if porcentajes is not None:
        array = array[array[:, 1].argsort(axis=1)]

    return array




def Retrieval_combined (list_imagenes,etiquetes_C,etiquetes_S,preguntaC,preguntaS,porcentajes_C=None,porcentajes_S=None):

    color=Retrieval_by_color (list_imagenes,etiquetes_C,preguntaC,porcentajes_C)

    shape = Retrieval_by_shape(list_imagenes,etiquetes_S,preguntaS,porcentajes_S)

    igualesL=[]

    for i in range(len(color)):
        for j in range(len(shape)):
            if np.array_equal(color[i],shape[j]):
                if porcentajes_C is not None and porcentajes_S is not None:
                    igualesL.append([color[i],porcentajes_S*porcentajes_S])
                else:
                    igualesL.append(color[i])

    iguales=np.array(igualesL)

    if porcentajes_C is not None and porcentajes_S is not None:
        iguales=iguales[iguales[:, 1].argsort(axis=1)]

    return iguales


def Kmean_statistics(Kmeans,Kmax):

    WCD = np.zeros(Kmax-1)
    num_iter=np.zeros(Kmax-1)

    for i in range(2,Kmax+1):
        Kmeans.K=i
        Kmeans.fit()
        WCD[i-2]=Kmeans.whitinClassDistance()
        num_iter[i-2]=Kmeans.num_iter

    plt.plot(range(2,Kmax+1),WCD, label="WCD")
    plt.plot(range(2,Kmax+1),num_iter, label="Num Iter")
    plt.legend()
    plt.xlabel('K')
    plt.show()


def Get_shape_accuracy(etiquetesKNN,GT):

    counter=float(0)
    for i in range(len(GT)):
        if etiquetesKNN[i]==GT[i]:
           counter+=1
    return (counter/len(etiquetesKNN))*100


def Get_color_accuracy(etiquetesKmeans,GT):
    counterT=float(0)
    for i in range(len(etiquetesKmeans)):
        GTcontador=float(0)

        for numero in range(len(GT[i])):

            coun=collections.Counter(etiquetesKmeans[i])

            GTcontador+=coun[GT[i][numero]]

        counterT+= (GTcontador/len(etiquetesKmeans[i]))

    return (counterT/len(etiquetesKmeans))*100

