import cv2 as cv
import numpy as np
import pdb

def convoluteWith1DMask(ker,horizontally,im):
    '''
    @brief Se convoluciona una imagen con una mascara 1 dimensional en una sola direccion horizontal
    o vertical.
    @param ker mascara con la que se hace la convolucion
    @param horizontally booleano que define si la convolucion se hace en horizontal o en vertical
    @param im numpy array de enteros sin signo que contiene informacion relativa a una imagen
    @return Devuelve un vector con todas los canales de la imagen convolucionados.
    '''
    # Definimos los nucleos de convolucion
    kerX = kerY = None
    if horizontally:
        # Si convolucionamos por filas el nucleo que no es trivial es kerX
        kerX = np.array(ker) # Reversed porque filter hace correlacion
        # Nucleo trivial [0,...,0,1,0,...,0]
        kerY = np.zeros(len(ker),dtype='int64')
        kerY[int(len(ker)/2)] = 1
    else:
        # Lo contrario
        # Nucleo trivial [0,...,0,1,0,...,0]
        kerX = np.zeros(len(ker),dtype='int64')
        kerX[int(len(ker)/2)] = 1
        # Si convolucionamos por filas el nucleo que no es trivial es kerY
        kerY = np.array(ker)
    if len(im.shape) == 3:      # Para imagenes RGB o LAB
        alto, ancho, profundo = im.shape
        # Se tratan los tres canales por separado
        inputSignals = cv.split(im)
        convolutedSignals = []
        for i in range(profundo):
            inputSignal = inputSignals[i]
            # Se convoluciona la senal R,G o B
            convolutedSignal = cv.sepFilter2D(inputSignal,-1,kerX,kerY)
            convolutedSignals.append(convolutedSignal)
        return convolutedSignals
    else:                       # Para imagenes GRAYSCALE
        return [cv.sepFilter2D(im,-1,kerX,kerY)]

def normaEuclidea(v):
    '''
    @brief Se calcula la norma euclidea de un vector
    @param im numpy array que contiene las coordenadas del vector
    @return Devuelve la norma de un vector
    '''
    return np.sqrt(np.dot(v,v))

def getGradient(signalsdx,signalsdy):
    '''
    @brief Calcula un gradiente a partir de las derivadas en las direcciones de x de
    todos los canales de una imagen
    @param signalsdx Derivada en la direccion de equis para todos los canales de una
    imagen
    @param signalsdy Derivada en la direccion de i para todos los canales de una
    imagen
    @return Devuelve el gradiente formado por en cada pixel el gradiente de mayor norma de entre todos
    los gradientes de todos los canales
    '''
    shape = None
    # Contient el gradiente final
    ret = []
    # Contiene los gradientes de todos los canales de la imagen
    gradientes = []
    for i in range(len(signalsdx)):
        dx = np.array(signalsdx[i])
        dy = np.array(signalsdy[i])
        shape = dx.shape
        dxprima = dx.reshape(-1)
        dyprima = dy.reshape(-1)
        # Formamos parejas de la forma (df/dx, df/dy)
        gradiente = np.array([[ex,ey] for ex, ey in np.transpose(np.vstack([dxprima,dyprima]))])
        gradientes.append(gradiente)
    N = len(gradientes[0])
    for i in range(N):
        normas = []
        for k in range(3):
            v = gradientes[k][i]
            f = normaEuclidea(v)
            normas.append(f)
        # Verificamos en cada pixel que
        indiceMax = np.argmax(normas, axis=None)
        ret.append(gradientes[indiceMax][i])
    ret = np.array(ret)
    ret = ret.reshape((shape[0],shape[1],2))
    return ret

def obtainAngle(vector):
    '''
    @brief Función que obtiene el ángulo del vector entre 0 y 180 grados
    @param vector Vector del que queremos obtener el ángulo
    @return Devuelve un valor real entre 0 y 180.
    '''
    angle_360=-1
    if vector[0]==0:
        angle_360 = 90
    elif vector[1]==0:
        angle_360 = 0
    else:
        angle_360 = np.rad2deg(np.arctan(vector[1]/vector[0]))

    angle = angle_360 if angle_360<=180 else angle_360-180
    return angle

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
            # Calculamos el ángulo
            angle = obtainAngle(gradient)
            # Obtenemos el floor y ceiling
            ceil = int(np.ceil(angle))
            floor = int(np.floor(angle))
            value = 1 if ceil==floor else np.absolute(angle-ceil)
            # Añadimos el valor de ceiling al histograma
            if not str(ceil) in histogram:
                histogram[str(ceil)] = value*normaEuclidea(gradient)
            else:
                histogram[str(ceil)]+=value*normaEuclidea(gradient)
            # Cuando el resultado del ángulo no es entero añadimos el floor también
            if ceil!=floor:
                if not str(floor) in histogram:
                    histogram[str(floor)] = np.absolute(angle-floor)*normaEuclidea(gradient)
                else:
                    histogram[str(floor)]+=np.absolute(angle-floor)*normaEuclidea(gradient)

    # Convertimos el diccionario a un vector
    for i in range(181):
        if not str(i) in histogram:
            histogram_vec.append(0)
        else:
            histogram_vec.append(histogram[str(i)])

    return histogram_vec

def modifyLocalMatrix(img,local_matrix,row_min,row_max,col_min,col_max):
    '''
    @brief Función que modifica la submatriz de img dada por los valores row_min,
    row_max, col_min y col_max con los valores de local_matrix
    @param img Matriz de la imagen que queremos modificar
    @param local_matrix Matriz local que tiene los valores que queremos poner en img
    @param row_min Valor mínimo en las filas (se empieza desde este índice)
    @param row_max Valor máximo en las filas (no se llega a tomar este valor de índice)
    @param col_min Valor mínimo en las columnas (se empieza desde este índice)
    @param col_max Valor máximo en las columnas (no se llega a tomar este valor de índice)
    @return Devuelve una copia de img con los valores modificados según local_matrix
    '''
    # Creamos una copia de la imagen
    img_aux = np.copy(img)
    for i in range(row_min,row_max):
        for j in range(col_min,col_max):
            # Modificamos los valores de la imagen auxiliar con los de local_matrix
            img_aux[i][j]=local_matrix[i-row_min][j-col_min]
    return img_aux
