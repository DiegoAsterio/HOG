import cv2 as cv
import numpy as np
import pdb

def convoluteWith1DMask(ker,enHorizontal,im):
    kerX = kerY = None

    if enHorizontal:
        kerX = np.array(ker)              # reversed(ker) (?) #Convolucion
        kerY = np.array([0,1,0])
    else:
        kerX = np.array([0,1,0])
        kerY = np.array(ker)              # reversed(ker) (?) #Convolucion
    alto = ancho = profundo = None
    if len(im.shape) == 3:      # Para imagenes RGB o LAB
        alto, ancho, profundo = im.shape
        inputSignals = cv.split(im)
        convolutedSignals = []
        for i in range(profundo):
            inputSignal = inputSignals[i]
            convolutedSignal = cv.sepFilter2D(inputSignal,-1,kerX,kerY)
            convolutedSignals.append(convolutedSignal)
        return convolutedSignals
    else:                       # Para imagenes GRAYSCALE
        return cv.sepFilter2D(im,-1,kerX,kerY)

def normaEuclidea(v):
    return np.sqrt(np.dot(v,v))    

def getGradient(signalsdx,signalsdy):
    ret = []
    gradientes = []
    for i in range(len(signalsdx)):
        dx = np.array(signalsdx[i])
        dy = np.array(signalsdy[i])
        shape = dx.shape
        dxprima = dx.reshape(-1)
        dyprima = dy.reshape(-1)
        gradiente = np.array([[ex,ey] for ex, ey in np.transpose(np.vstack([dxprima,dyprima]))])
        gradiente = gradiente.reshape((shape[0],shape[1],2))
        gradientes.append(gradiente)
    ancho, alto, prof = gradientes[0].shape
    for i in range(ancho):
        for j in range(alto):
            normas = []
            for k in range(3):
                v = gradientes[k][i][j]
                f = normaEuclidea(v)
                normas.append(f)
            normas = np.array(normas)
            indiceMax = np.argmax(normas, axis=None)
            ret.append(gradientes[indiceMax][i][j])
    return ret
