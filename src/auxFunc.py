import cv2 as cv
import numpy as np
import pdb

def convoluteWith1DMask(ker,enHorizontal,im):
    kerX = kerY = None

    if enHorizontal:
        kerX = np.array(ker)              # reversed(ker) (?) #Convolucion
        kerY = np.zeros(len(ker))
        kerY[len(ker)/2] = 1
    else:
        kerX = np.zeros(len(ker))
        kerX[len(ker)/2] = 1
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

def computeHistogram(cell):
    '''
    @brief Dada una célula con un vector gradiente en cada posición coge el ángulo
    de cada vector y hace un histograma en forma de vector con los ángulos ponderados.
    @param cell Matriz con los datos del gradiente que representa una célula
    @return Devuelve un vector  de 180 elementos donde tiene un 0 si el ángulo no aparece
    o un valor correspondiente a la interpolación bilineal al obtener el histograma.
    '''
    # Inicializamos el histograma
    histogram = {}
    histogram_vec = []

    # Para cada elemento del histograma
    for row in cell:
        for gradient in row:
            # Calculamos su ángulo entre 0 y 360
            angle_360 = np.rad2deg(np.arctan(gradient[1]/gradient[0]))
            # Lo reducimos a [0,180]
            angle = angle_360 if angle_360<180 else angle_360-180
            # Obtenemos el floor y ceiling
            ceil = np.ceil(angle)
            floor = np.floor(angle)
            value = 1 if ceil==floor else np.absolute(angle-ceil)
            # Añadimos el valor de ceiling al histograma
            if not str(ceil) in histogram:
                histogram[ceil] = value*normaEuclidea(gradient)
            else:
                histogram[ceil]+=value*normaEuclidea(gradient)
            # Cuando el resultado del ángulo no es entero añadimos el floor también
            if ceil!=floor:
                if not str(floor) in histogram:
                    histogram[floor] = np.absolute(angle-floor)*normaEuclidea(gradient)
                else:
                    histogram[floor]+=np.absolute(angle-floor)*normaEuclidea(gradient)

    # Convertimos el diccionario a un vector
    for i in range(181):
        if not str(i) in histogram:
            histogram_vec.append(0)
        else
            histogram_vec.append(histogram[str(i)])

    return histogram_vec
