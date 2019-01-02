import cv2 as cv
import numpy as np
import pdb
import os
import re

PATH_TO_INRIA = "../../INRIAPerson"

################################################################################
##                       Funciones de dibujado                                ##
################################################################################

def pintaMI(vim):
    imagenes = []
    max_h = 0
    max_w = 0
    for im in vim:
        if im.shape[0]>max_h:
            max_h = im.shape[0]
        if im.shape[1]>max_w:
            max_w = im.shape[1]
    for im in vim:
        if len(im.shape)==2:
            imagenes.append(cv.copyMakeBorder(cv.cvtColor(im,cv.COLOR_GRAY2RGB),top=0,bottom=max_h-im.shape[0],left=0,right=0,borderType= cv.BORDER_CONSTANT, value=[0,0,0]))
        else:
            imagenes.append(cv.copyMakeBorder(im,top=0,bottom=max_h-im.shape[0],left=0,right=0,borderType= cv.BORDER_CONSTANT, value=[0,0,0]))
    concatenada = cv.hconcat(imagenes)
    cv.namedWindow('Imagenes', cv.WINDOW_NORMAL)
    cv.imshow("Imagenes",concatenada)
    cv.waitKey(0)
    cv.destroyAllWindows()


################################################################################
##                        Preprocesamiento                                    ##
################################################################################

def parseAnnotationFile(img_file_name,annotation_dir):
    '''
    @brief Función que devuelve un diccionario con la información de la imagen pasada como argumento
    @param img_file_name Ruta de la imagen de la que queremos obtener información
    @param annotation_dir Ruta del fichero de anotaciones correspondiente
    @return Devuelve un diccionario que contiene el tamaño de la imagen,
    el número de peatones y una lista que delimita la ventana en la que está el peatón
    '''
    # Ruta al txt con las anotaciones
    annotation_path = annotation_dir + os.path.splitext(img_file_name)[0] + '.txt'

    # Abrimos el fichero de anotaciones
    f = open(annotation_path)
    lines = f.readlines()
    f.close()

    # Recopilamos la información
    object_list = []
    object_info = {}
    ground_truth = None
    img_size = None

    for line in lines:
        # Cogemos el tamaño de la imagen
        m = re.match(r'Image size \(X x Y x C\) : (\d+) x (\d+) x 3', line)
        if m:
            img_size = (int(m.group(1)), int(m.group(2)))

        # El número de peatones
        m = re.match(r'Objects with ground truth : (\d+)', line)
        if m:
            ground_truth = int(m.group(1))

        if line.find('# Details for object') != -1:
            object_info = {}

        # Buscamos el centro
        m = re.match(r'Center point on object (\d)+ "PASperson" \(X, Y\) : \((\d+), (\d+)\)', line)
        if m:
            center = (int(m.group(2)), int(m.group(3)))
            object_info['center'] = center

        # Obtenemos la caja que delimita el peatón
        m = re.match(r'Bounding box for object (\d+) "PASperson" \(Xmin, Ymin\) - \(Xmax, Ymax\) : \((\d+), (\d+)\) - \((\d+), (\d+)\)', line)
        if m:
            bounding_box = [(int(m.group(2)), int(m.group(3))), (int(m.group(4)), int(m.group(5)))]
            object_info['bounding_box'] = bounding_box
            object_list.append(object_info)

    # Comprobamos el número de cajas obtenidas para que coincida con el de peatones.
    if len(object_list) != ground_truth:
        Exception("El número de peatones y cajas obtenidas no concuerda")
        return None

    # Creamos el diccionario
    annotation_info = {
        'img_size': img_size,
        'ground_truth': ground_truth,
        'object_list': object_list
    }

    return annotation_info

################################################################################
##                         Funciones de cálculo                               ##
################################################################################

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

def convexCombOfTwo(point, vpoints):
    for i in range(1,len(vpoints)):
        if vpoints[i]>point:
            tam = vpoints[1]-vpoints[0]
            coef1 = 1 - (point-vpoints[i-1])/tam
            coef2 = vpoints[i]-point/tam
            return i-1, coef1, i, coef2
    return False

def computeHistogramDiego(cell, num_cols, threeSixtyQ=False):
    possibleAngles = []
    histogram = np.zeros(num_cols)
    if threeSixtyQ:
        possibleAngles = np.linspace(0,360,num_cols)
    else:
        possibleAngles = np.linspace(0,180,num_cols)
    for row in cell:
        for gradient in row:
            angle = obtainAngle(gradient)
            indice1, coef1, indice2, coef2 = convexCombOfTwo(angle,possibleAngles)
            voto = normaEuclidea(gradient)
            histogram[indice1] += coef1*voto
            histogram[indice2] += coef2*voto
    return list(histogram)

def computeHistogram(cell,num_cols):
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
    @param row_max Valor máximo en las filas (no se tinta en inglesllega a tomar este valor de índice)
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

def loadTrainImgs():
    '''
    @brief Función que devuelve las imágenes de entrenamiento como dos listas
    @return Devuelve dos vectores, el primero con los casos positivos, es decir,
    donde si hay personas y el segundo con los casos negativos, es decir, donde
    no hay personas.
    '''
    pos_imgs = []
    neg_imgs = []
    pos_imgs_names = os.listdir(PATH_TO_INRIA+"/Train/pos")
    for pimg in pos_imgs_names:
        pos_imgs.append(cv.imread(PATH_TO_INRIA+"/Train/pos/"+pimg,-1))
    neg_imgs_names = os.listdir(PATH_TO_INRIA+"/Train/neg")
    for nimg in neg_imgs_names:
        neg_imgs.append(cv.imread(PATH_TO_INRIA+"/Train/neg/"+nimg,-1))
    return pos_imgs,neg_imgs

def loadTestImgs():
    '''
    @brief Función que devuelve las imágenes de test como dos listas
    @return Devuelve dos vectores, el primero con los casos positivos, es decir,
    donde si hay personas y el segundo con los casos negativos, es decir, donde
    no hay personas.
    '''
    pos_imgs = []
    neg_imgs = []
    pos_imgs_names = os.listdir(PATH_TO_INRIA+"/Test/pos")
    for pimg in pos_imgs_names:
        pos_imgs.append(cv.imread(PATH_TO_INRIA+"/Test/pos/"+pimg,-1))
    neg_imgs_names = os.listdir(PATH_TO_INRIA+"/Test/neg")
    for nimg in neg_imgs_names:
        neg_imgs.append(cv.imread(PATH_TO_INRIA+"/Test/neg/"+nimg,-1))
    return pos_imgs,neg_imgs
