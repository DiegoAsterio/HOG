import cv2 as cv
import numpy as np
import pdb
import os
import re
import random

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
##                          Preprocesamiento                                  ##
################################################################################

def obtainCropLimits(nrows,ncols,window_size=(64,128)):
    x0 = random.randint(0,nrows)
    y0 = random.randint(0,ncols)
    while x0+window_size[0]>=ncols or y0+window_size[1]>=nrows:
        x0 = random.randint(0,nrows)
        y0 = random.randint(0,ncols)
    x1 = x0+window_size[0]
    y1 = y0+window_size[1]
    return x0,y0,x1,y1

def obtainNegativeSamples(neg_samples_dir="../../INRIAPerson/Train/neg/",dir_to_save="./cropped_neg/"):
    list_images = os.listdir(neg_samples_dir)
    for img_name in list_images:
        img = cv.imread(neg_samples_dir + img_name,-1)
        img_name_sp = img_name.split(".")[0]
        format = img_name.split(".")[1]
        for i in range(10):
            x_min,y_min,x_max,y_max = obtainCropLimits(img.shape[0],img.shape[1])
            crop = img[y_min:y_max, x_min:x_max]
            cv.imwrite(dir_to_save+img_name_sp+"_c_"+str(i)+"."+format,crop)

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
        # Reversed porque sepFilter2D hace correlacion
        kerX = np.array(list(reversed(ker)))
        # Nucleo trivial [...,0,1,0,...]
        kerY = np.array([1])
    else:
        # Analogamente
        # Nucleo trivial [...,0,1,0,...]
        kerX = np.array([1])
        # Si convolucionamos por filas el nucleo que no es trivial es kerY
        # Reversed porque sepFilter2D hace correlacion
        kerY = np.array(list(reversed(ker)))
    if len(im.shape) == 3:      # Para imagenes RGB o LAB
        alto, ancho, profundo = im.shape
        # Se tratan los tres canales por separado
        inputSignals = cv.split(im)
        convolutedSignals = []
        for i in range(profundo):
            inputSignal = inputSignals[i]
            # Se convoluciona las senales R,G y B
            convolutedSignal = cv.sepFilter2D(inputSignal,-1,kerX,kerY)
            convolutedSignals.append(convolutedSignal)
        return convolutedSignals
    else:
        # Para imagenes GRAYSCALE
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
    n,m = signalsdx[0].shape
    # Contiene el gradiente final
    ret = []
    # Contiene los gradientes de todos los canales de la imagen
    gradientes = []
    for i in range(len(signalsdx)):
        dx = np.array(signalsdx[i]).reshape(-1)
        dy = np.array(signalsdy[i]).reshape(-1)
        # Formamos parejas de la forma (df/dx, df/dy)
        gradiente = np.stack((dx,dy),axis=1)
        gradientes.append(gradiente)
    normasGradientes = [gradiente**2 for gradiente in gradientes]
    for i in range(n*m):
        normas = []
        for k in range(3):
            xx = normasGradientes[k][i,0]
            yy = normasGradientes[k][i,1]
            normas.append(xx+yy)
        # Verificamos en cada pixel que
        indiceMax = np.argmax(normas, axis=None)
        ret.append(gradientes[indiceMax][i])
    ret = np.array(ret)
    ret = ret.reshape((n,m,2))
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
    '''
    @brief Dada una célula con un vector gradiente en cada posición coge el ángulo
    de cada vector y hace un histograma en forma de vector con los ángulos ponderados.
    @param cell Matriz con los datos del gradiente que representa una célula
    @return Devuelve un vector  de 180 elementos donde tiene un 0 si el ángulo no aparece
    o un valor correspondiente a la interpolación bilineal al obtener el histograma.
    '''
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
    pos_imgs_names = os.listdir(PATH_TO_INRIA+"/cropped_pos")
    for pimg in pos_imgs_names:
        im = cv.imread(PATH_TO_INRIA+"/cropped_pos/"+pimg,-1)
        #im = np.float32(im)
        pos_imgs.append(im)
    neg_imgs_names = os.listdir(PATH_TO_INRIA+"/cropped_neg")
    for nimg in neg_imgs_names:
        neg_imgs.append(cv.imread(PATH_TO_INRIA+"/cropped_neg/"+nimg,-1))
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
        im = cv.imread(PATH_TO_INRIA+"/Test/pos/"+pimg,-1)
        #im = np.float32(im)
        pos_imgs.append(im)
    neg_imgs_names = os.listdir(PATH_TO_INRIA+"/Test/neg")
    for nimg in neg_imgs_names:
        neg_imgs.append(cv.imread(PATH_TO_INRIA+"/Test/neg/"+nimg,-1))
    return pos_imgs,neg_imgs

def getAllWindows(im,window_size=(64,128)):
    '''
    @brief Función que devuelve todas las submatrices de 64x128 de la imagen im
    @param im Imagen de la que queremos sacar las submatrices
    @param window_size Tupla que nos da las dimensiones de la ventana
    @return Devuelve una lista con las submatrices de 64x128 extraídas
    '''
    ret = []
    m = im.shape[0]
    n = im.shape[1]
    for i in range(m-window_size[0]):
        for j in range(n-window_size[1]):
            ret.append(im[i:i+window_size[0],j:j+window_size[1]])
    return ret
