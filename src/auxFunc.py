import cv2 as cv
import numpy as np
import pdb
import os
import re
import random
import descriptorHOG
from profilehooks import profile
from numba import jit, autojit

PATH_TO_INRIA = "../INRIAPerson"

GLOBAL_COUNT=0

################################################################################
##                       Funciones de dibujado                                ##
################################################################################

def pintaMI(vim):
    '''
    @brief Función que dada una lista de imágenes las pinta en una misma ventana.
    @param vim Lista de imágenes que queremos pintar
    '''
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
    '''
    @brief Función que dados el número de filas y columnas de la imagen original
    da los límites de una ventana aleatoria de tamaño window_size
    @param nrows Número de filas de la imagen original
    @param ncols Número de columnas de la imagen original
    @param window_size Tamaño de la ventana que queremos obtener, por defecto es
    de tamaño 64x128
    @return Devuelve las esquinas con valores mínimos y máximo de x e y
    '''
    x0 = random.randint(0,nrows)
    y0 = random.randint(0,ncols)
    while x0+window_size[0]>=ncols or y0+window_size[1]>=nrows:
        x0 = random.randint(0,nrows)
        y0 = random.randint(0,ncols)
    x1 = x0+window_size[0]
    y1 = y0+window_size[1]
    return x0,y0,x1,y1

def obtainNegativeSamples(neg_samples_dir=PATH_TO_INRIA+"/Train/neg/",dir_to_save="./cropped_neg/"):
    '''
    @brief Función que dado un directorio con imágenes y un directorio para guardarlas
    obtiene 10 ventanas aleatorias de la misma y las guarda en el directorio correspondiente
    @param neg_samples_dir Directorio que contiene las imágenes
    @param dir_to_save Directorio donde queremos guardar los resultados
    '''
    list_images = os.listdir(neg_samples_dir)
    for img_name in list_images:
        img = cv.imread(neg_samples_dir + img_name,-1)
        img_name_sp = img_name.split(".")[0]
        format = img_name.split(".")[1]
        for i in range(10):
            x_min,y_min,x_max,y_max = obtainCropLimits(img.shape[0],img.shape[1])
            crop = img[y_min:y_max, x_min:x_max]
            cv.imwrite(dir_to_save+img_name_sp+"_c_"+str(i)+"."+format,crop)

def obtainNegatives(imgs,svm,neg_samples_dir=PATH_TO_INRIA+"/Train/neg/", num_windows=10):
    '''
    @brief Función que dado un directorio con imágenes y un directorio para guardarlas
    obtiene 10 ventanas aleatorias de la misma y las guarda en el directorio correspondiente
    @param neg_samples_dir Directorio que contiene las imágenes
    @param dir_to_save Directorio donde queremos guardar los resultados
    '''
    pred = []
    contador=1
    for img in imgs:
        print("Obteniendo predicción de la imagen " + str(contador) + "/" + str(len(imgs)))
        contador+=1
        windows = []
        for i in range(num_windows):
            x_min,y_min,x_max,y_max = obtainCropLimits(img.shape[0],img.shape[1])
            crop = img[y_min:y_max, x_min:x_max]
            windows.append(crop)
        descr = descriptorHOG.obtainDescriptors(windows)
        pred_windows = svm.predict(descr)[1]
        pred_windows = [pred[0] for pred in pred_windows]
        pred.append(pred_windows)
    return pred

def obtainHardNegativeExamples(svm, hard_training_dir=PATH_TO_INRIA+"/hard_negative_examples/"):
    negatives = obtainNegativesRaw()
    names = [name.split(".")[0] for name in os.listdir(PATH_TO_INRIA+"/Train/neg/")]
    formats = [name.split(".")[1] for name in os.listdir(PATH_TO_INRIA+"/Train/neg/")]
    contador = 1
    for i in range(len(negatives)):
        print("Encontrando ejemplos dificiles "+str(contador)+"/"+str(len(negatives)))
        contador += 1
        # Es una lista de listas en la que en cada posición tiene las pirámides gaussianas de cada imagen en la ventana
        pyr = gaussianPyramid(negatives[i])
        windows=[]
        for level in pyr:
            y,x,z = level.shape
            indiceX = 0
            indiceY = 0
            # Comprobamos si nos salimos de los límites de la imagen
            while indiceY+128<y:
                indiceX = 0
                while indiceX+64<x:
                    # Tomamos el crop del subnivel
                    windows.append(level[indiceY:indiceY+128,indiceX:indiceX+64])
                    indiceX = indiceX + 20
                indiceY = indiceY + 20
        # Obtenemos los descriptores
        descr = descriptorHOG.obtainDescriptors(windows,True)
        # Predecimos y damos formato a las predicciones
        predicted = svm.predict(descr)[1]
        predicted = [pred[0] for pred in predicted]
        # Contador para el nombre
        fail_count = 0
        for j in range(len(predicted)):
            if predicted[j]==1:
                cv.imwrite(hard_training_dir+names[i]+"_image"+str(i)+"_failed"+str(fail_count)+"."+formats[i],np.uint8(windows[j]))
                fail_count+=1

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
    return np.linalg.norm(v)

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
    dx = []
    dy = []
    # Contiene los gradientes de todos los canales de la imagen
    cuadradoX = [gx**2 for gx in signalsdx]
    cuadradoY = [gy**2 for gy in signalsdy]
    for i in range(n):
        for j in range(m):
            normas = []
            for k in range(3):
                xx = cuadradoX[k][i,j]
                yy = cuadradoY[k][i,j]
                normas.append(xx+yy)
            # Verificamos en cada pixel que
            indiceMax = np.argmax(normas, axis=None)
            dx.append(signalsdx[indiceMax][i,j])
            dy.append(signalsdy[indiceMax][i,j])
    dx = np.array(dx)
    dy = np.array(dy)
    dx = dx.reshape((n,m))
    dy = dy.reshape((n,m))
    return dx, dy

def convexCombOfTwo(point, vpoints):
    '''
    @brief Función que dado un ángulo y una lista de ángulos posibles nos dice
    en qué porcentaje debemos sumar a cada índice del histograma
    @param point Ángulo del que queremos obtener los coeficientes
    @param vpoints Lista que contiene una división equiespaciada de los posibles
    valores del ángulo point
    @return Devuelve una 4-upla que contiene el valor del primer índice, su
    coeficiente correspondiente, el valor del segundo índice y su coeficiente
    correspondiente
    '''
    N = len(vpoints)
    for i in range(1,N):
        # En el momento en que encontremos el ángulo de la lista que es mayor que el nuestro
        if vpoints[i]>point:
            # Calculamos los coeficientes y los devolvemos
            tam = vpoints[1]-vpoints[0]
            coef1 = 1 - (point-vpoints[i-1])/tam
            coef2 = (point-vpoints[i-1])/tam
            return i-1, coef1, i, coef2
    # Si el ángulo es justo el último asignamos los coeficientes
    if point == vpoints[N-1]:
        return N-2, 0, N-1, 1
    return False

def computeHistogram(subMag, subAng, num_cols, threeSixtyQ=False):
    '''
    @brief Dada una célula con un vector gradiente en cada posición coge el ángulo
    de cada vector y hace un histograma en forma de vector con los ángulos ponderados.
    @param cell Matriz con los datos del gradiente que representa una célula
    @return Devuelve un vector  de 180 elementos donde tiene un 0 si el ángulo no aparece
    o un valor correspondiente a la interpolación bilineal al obtener el histograma.
    '''
    m,n = subMag.shape
    possibleAngles = []
    # Inicializamos el histogama
    histogram = np.zeros(num_cols)
    # Si estamos en 0,360 o en 0,180 hacemos la lista de posibles ángulos
    if threeSixtyQ:
        possibleAngles = np.linspace(0,360,num_cols)
    else:
        possibleAngles = np.linspace(0,180,num_cols)
    # Para cada posición calculamos el ángulo y sumamos el valor proporcional a la
    # magnitud a la posición correspondiente del histograma
    for i in range(m):
        for j in range(m):
            mag = subMag[i,j]
            angle = None
            if threeSixtyQ:
                angle = subAng[i,j]
            else:
                angle = subAng[i,j] if subAng[i,j] < 180 else subAng[i,j]-180
            indice1, coef1, indice2, coef2 = convexCombOfTwo(angle,possibleAngles)
            histogram[indice1] += coef1*mag
            histogram[indice2] += coef2*mag
    return list(histogram)

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

def getReflectedImage(img):
    '''
    @brief Función que devuelve la imagen reflejada con respecto al borde derecho de
    la imagen img
    @param img Imagen de la que queremos obtener la reflejada
    @return Devuelve una imagen del mismo tamaño que representa la reflejada de la original
    '''
    # Obtenemos la  imagen reflejada
    return np.flip(img,axis=1)

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
        im = np.float32(im)
        im_reflected = getReflectedImage(im)
        pos_imgs.append(im)
        pos_imgs.append(im_reflected)
    neg_imgs_names = os.listdir(PATH_TO_INRIA+"/cropped_neg")
    for nimg in neg_imgs_names:
        im = cv.imread(PATH_TO_INRIA+"/cropped_neg/"+nimg,-1)
        im = np.float32(im)
        neg_imgs.append(im)
    return pos_imgs,neg_imgs

def loadHardNegativeExamples():
    '''
    @brief Funcion que devuelve las imagenes que son ejemplos
    dificiles
    @return Lista que contiene imagenes
    '''
    vim = []
    hard_examples_names = os.listdir(PATH_TO_INRIA+"/hard_negative_examples")
    for pimg in hard_examples_names:
        im = cv.imread(PATH_TO_INRIA+"/hard_negative_examples/"+pimg,-1)
        im = np.float32(im)
        vim.append(im)
    return vim

def loadHardPositiveExamples(nImgs=-1):
    '''
    @brief Funcion que devuelve las imagenes que son ejemplos
    dificiles
    @return Lista que contiene imagenes
    '''
    vim = []
    hard_examples_names = os.listdir(PATH_TO_INRIA+"/hard_positive_examples")
    if nImgs>0:
        print("Tomando solo " + str(nImgs) + " imagenes dificiles positivas para entrenar")
        hard_examples_names = random.sample(hard_examples_names,nImgs)
    for pimg in hard_examples_names:
        im = cv.imread(PATH_TO_INRIA+"/hard_positive_examples/"+pimg,-1)
        im = np.float32(im)
        vim.append(im)
    return vim

def loadTestImgs():
    '''
    @brief Función que devuelve las imágenes de test como dos listas
    @return Devuelve dos vectores, el primero con los casos positivos, es decir,
    donde si hay personas y el segundo con los casos negativos, es decir, donde
    no hay personas.
    '''
    pos_imgs = []
    neg_imgs = []
    pos_imgs_names = os.listdir(PATH_TO_INRIA+"/cropped_test_pos")
    for pimg in pos_imgs_names:
        im = cv.imread(PATH_TO_INRIA+"/cropped_test_pos/"+pimg,-1)
        im = np.float32(im)
        pos_imgs.append(im)
    neg_imgs_names = os.listdir(PATH_TO_INRIA+"/cropped_test_neg")
    for nimg in neg_imgs_names:
        im = cv.imread(PATH_TO_INRIA+"/cropped_test_neg/"+nimg,-1)
        im = np.float32(im)
        neg_imgs.append(im)
    return pos_imgs,neg_imgs

def obtainNegativesRaw():
    vim = []
    neg_imgs_names = os.listdir(PATH_TO_INRIA+"/Train/neg")
    for imname in neg_imgs_names:
        im = cv.imread(PATH_TO_INRIA+"/Train/neg/"+imname)
        im = np.float32(im)
        vim.append(im)
    return vim

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
    for i in range(m-window_size[1]):
        for j in range(n-window_size[0]):
            ret.append(im[i:i+window_size[1],j:j+window_size[0]])
    return ret


def gaussianPyramid(img,levels=3):
    '''
    @brief Esta función obtiene la pirámide gaussiana de la imagen dada.
    @param img Imagen a la que se le quiere calcular la pirámide gaussiana.
    @param levels Niveles de la pirámide gaussiana, por defecto 3.
    @return Devuelve la pirámide gaussiana de la imagen pasada.
    '''
    pyr = []
    # Se hace un downsample a la imagen. La función pyrDown implementa ya el blur.
    img_pyr = cv.pyrDown(img)
    pyr.append(img)
    pyr.append(img_pyr)
    # Se hace el downsample y el blur tantas veces como niveles se quieran a la imagen una y otra vez.
    for i in range(levels-2):
        img_pyr = cv.pyrDown(img_pyr)
        pyr.append(img_pyr)
    return pyr

def getPedestrianBoxes(img_name,path_to_annotations):
    '''
    @brief Función que dado el nombre de una imagen de la carpeta Test de INRIAPerson
    por ejemplo crop001501.png devuelve las cajas en las que hay peatones
    @param img_name Nombre de la imagen sobre la que queremos obtener las cajas de los peatones
    @return Devuelve una lista de listas en las que en cada sublista se encuentran
    los valores xmin, ymin, xmax, ymax que delimitan el rectángulo del peatón
    '''
    # Nos quedamos sólo con el nombre crop001501
    name_only = img_name.split(".")[0]
    # Leemos el fichero, en ISO porque si no da fallos de procesamiento
    annotations_f = open(PATH_TO_INRIA+path_to_annotations+name_only+".txt","r",encoding = "ISO-8859-1")
    # Inicializamos la lista de cajas
    boxes = []
    for line in annotations_f:
        # Si en la línea se da información sobre la caja de peatones
        if "Bounding box" in line:
            # Magia y saco los valores, mejor no preguntes
            xmin = int(line.split(":")[1].split("-")[0].replace(" ","").split(",")[0].split("(")[-1])
            ymin = int(line.split(":")[1].split("-")[0].replace(" ","").split(",")[1].split(")")[0])
            xmax = int(line.split("-")[-1].replace(" ","").split(",")[0].split("(")[-1])
            ymax = int(line.split("-")[-1].replace(" ","").split(",")[1].split(")")[0])
            # Se une la lista a boxes
            boxes.append([xmin,ymin,xmax,ymax])
    return boxes

def getPredPos(imgs,boxes,svm):
    '''
    @brief Función que dado un vector de imágenes positivas (con peatones en ellas)
    y una lista de cajas que delimitan la posición de los peatones obtiene ventanas
    dentro de dicha imagen de tamaño 128x64 que solapan con el rectángulo del peatón.
    @param imgs Vector de imágenes
    @param boxes Lista de listas que contiene la delimitación del rectángulo del peatón
    @return Devuelve una lista de listas en las que cada posición tiene una lista
    para la imagen correspondiente de ventanas de 128x64
    '''
    boxes_pred = []
    for i in range(len(imgs)):
        print("Obteniendo predicción de la imagen " + str(i+1) + "/" + str(len(imgs)))
        box_pred= getPredPosImg(svm,imgs[i],boxes[i])
        boxes_pred.append(box_pred)
    return boxes_pred

def getPredNeg(svm,imgs):
    '''
    @brief Función que obtiene las predicciones de las imágenes negativas (no tienen peatones)
    @param imgs Lista de imágenes
    @return Devuelve una lista de listas en la que en cada posición tiene una lista
    de ventanas para la imagen correspondiente
    '''
    pred = obtainNegatives(svm,imgs,neg_samples_dir=PATH_TO_INRIA+"/Test/neg/")
    return pred


def checkArea(x1,y1,x2,y2,u1,v1,u2,v2):
    '''
    @brief Función que dados dos rectángulos comprueba si el segundo de ellos tiene
    al menos un 50% del área del primero.
    @param x1 Indice inferior en el eje X del rectángulo grande
    @param y1 Indice inferior en el eje Y del rectángulo grande
    @param x2 Indice superior en el eje X del rectángulo grande
    @param y2 Indice superior en el eje Y del rectángulo grande
    @param u1 Indice inferior en el eje X del rectángulo pequeño
    @param v1 Indice inferior en el eje Y del rectángulo pequeño
    @param u2 Indice superior en el eje X del rectángulo pequeño
    @param v2 Indice superior en el eje Y del rectángulo pequeño
    @return Devuelve un valor booleano que indica si el rectángulo delimitado por
    u1,v1,u2,v2 tiene al menos el 50% del área del rectángulo delimitado por
    x1,y1,x2,y2.
    '''
    areaTotal = float((x2-x1)*(y2-y1))
    areaParcial = float((u2-u1)*(v2-v1))
    return areaParcial/areaTotal >= 0.5

# Originalmente stepY=64, stepX=32
#def getPredPosImg(svm, img, boxes, stepY=16, stepX=8):
def getPredPosImg(svm, img, boxes, stepY=16, stepX=8):
    '''
    @brief Función que dada una imagen y la delimitación de sus peatones obtiene
    todas las ventanas que al menos solapan un 50% con los rectángulos de los peatones.
    El proceso se hace para todos los niveles de la pirámide Gaussiana
    @param img Imagen de la que queremos obtener las ventanas
    @param boxes Lista de listas que nos da la delimitación de los peatones en la imagen
    @param stepY Cantidad de píxeles en los que se avanza para obtener la siguiente ventana
    en el eje X de la imagen
    @param stepX Cantidad de píxeles en los que se avanza para obtener la siguiente ventana
    en el eje Y de la imagen
    @return Devuelve una lista de subimagenes de 128x64 para la imagen pasada
    '''
    global GLOBAL_COUNT

    imgs_con_boxes = []

    ret = np.zeros(len(boxes)).astype(np.bool)
    # Calculamos la pirámide gaussiana, la primera imagen de la misma es la original
    pyr = gaussianPyramid(img)
    scale=1
    # Para cada nivel
    for level in pyr:
        print("Escala:" + str(scale))
        windows=[]
        coord = []
        y,x,z = level.shape
        indiceY = indiceX = 0
        # Comprobamos si nos salimos de los límites de la imagen
        while indiceY+128<y:
            indiceX = 0
            while indiceX+64<x:
                windows.append(level[indiceY:indiceY+128,indiceX:indiceX+64])
                coord.append((indiceY, indiceX))
                indiceX = indiceX + stepX//scale
            indiceY = indiceY + stepY//scale
        if scale==4:
            # Lo meto dos veces por si windows esta vacia
            windows.append(cv.resize(level,(128,64)))
            windows.append(cv.resize(level,(128,64)))
        if len(windows)>1:
            print("Obteniendo descriptores")
            print("Número de ventanas: " + str(len(windows)))
            if len(windows)>4000:
                descr = descriptorHOG.obtainDescriptors(windows[:int(len(windows)/2)],silent=True)
                descr = np.concatenate((descr, descriptorHOG.obtainDescriptors(windows[int(len(windows)/2):],silent=True)))
            else:
                descr = descriptorHOG.obtainDescriptors(windows,silent=True)
            print("Obteniendo predicciones")
            prediction = [pred[0] for pred in svm.predict(descr)[1]]
            print("Obteniendo el mapa de calor")
            heatMap = buildHeatMap((y,x),prediction,coord)
            print("Obteniendo las ocurrencias")
            answer, boxes_nuestras = checkOccurrences(heatMap, boxes, scale)
            ret |= answer

            img_rectangulos=np.uint8(level)
            if len(boxes_nuestras)>0:
                cv.rectangle(img_rectangulos, (boxes_nuestras[0][0],boxes_nuestras[0][3]), (boxes_nuestras[0][2], boxes_nuestras[0][1]), (0,255,0), 3)
                for xmin,ymin,xmax,ymax in boxes_nuestras[1:]:
                    cv.rectangle(img_rectangulos, (xmin,ymax), (xmax, ymin), (0,255,0), 3)
                imgs_con_boxes.append(img_rectangulos)

        # Escalamos para obtener las coordenadas adecuadas en cada nivel de la pirámide Gaussiana
        scale*=2

    if len(imgs_con_boxes)>0:
        #pintaMI(imgs_con_boxes)
        for img_box in imgs_con_boxes:
            cv.imwrite("./cuadrados/"+str(GLOBAL_COUNT)+".jpg",img_box)
            GLOBAL_COUNT+=1
    else:
        print("No hay ninguna imagen con cajas")

    return ret

@autojit
def buildHeatMap(size, prediction, coord):
    HeatMap = np.zeros(size)
    for i in range(len(prediction)):
        if prediction[i] == 1:
            y, x = coord[i]
            HeatMap[y:y+128,x:x+64] += 1
    return HeatMap

@autojit
def differentFromZero(heatMap):
    x,y = np.nonzero(heatMap)
    indexes = list(zip(x,y))
    return indexes

@autojit
def cutBeneathRate(rate, heatMap):
    heatMap[heatMap<rate]=0
    return heatMap

@autojit
def checkOccurrences(heatMap, boxes, scale):
    umbral = 1      # Es como si no hubiese umbral (<1 es cero)
    m = cutBeneathRate(umbral,heatMap)
    indexes = differentFromZero(m)
    regions = []
    while len(indexes) != 0:
        region = getRegion(indexes[0],heatMap,indexes)
        indexes = substractRegion(region,indexes)
        regions.append(region)
    ourBoxes = createBoxes(regions,scale,heatMap)
    answer = np.zeros(len(boxes)).astype(np.bool)
    for xmin, ymin, xmax, ymax in ourBoxes:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            if checkArea(x1//scale, y1//scale, x2//scale, y2//scale, xmin, ymin, xmax, ymax):
                answer[i] = True
    return answer, ourBoxes

@autojit
def createBoxes(regions,scale,G):
    boxes = []
    y_boundary,x_boundary = G.shape
    for region in regions:
        subRegion = getElemsMax(G,region)
        x, y = getCenter(subRegion)
        #boxes.append((x-32//scale, y-64//scale, x+32//scale, y+64//scale))
        boxes.append(obtainBox(x,y,y_boundary, x_boundary))
    return boxes

@autojit
def obtainBox(x,y,x_boundary,y_boundary):
    xmin = 0 if x-32<0 else x-32
    ymin = 0 if y-64<0 else y-64
    xmax = x_boundary-1 if x+32>=x_boundary else x+32
    ymax = y_boundary-1 if y+64>=y_boundary else y+64
    return (xmin,ymin,xmax,ymax)

@autojit
def getElemsMax(G,region):
    maxim = -1
    for i,j in region:
        if maxim < G[i,j]:
            maxim = G[i,j]
    ret = []
    for i,j in region:
        if G[i,j] == maxim:
            ret.append((i,j))
    return ret

def getCenter(region):
    xmin = min(list(map(lambda x:x[0],region)))
    ymin = min(list(map(lambda x:x[1],region)))
    xmax = max(list(map(lambda x:x[0],region)))
    ymax = max(list(map(lambda x:x[1],region)))
    x = int((xmin + xmax)/2)
    y = int((ymin + ymax)/2)
    return x, y

@autojit
def getRegion(index, G,indexes):
    indices = indexes.copy()
    region = [index]
    non_tested = [index]
    while non_tested:
        ind_reg = non_tested.pop()
        for ind in indices:
            if checkAdjacent(ind_reg,ind):
                region.append(ind)
                non_tested.append(ind)
                indices.remove(ind)
    return list(set(region))

@autojit
def checkAdjacent(i1,i2):
    return ((i1[0]==i2[0]+1 or i1[0]==i2[0]-1) and i1[1]==i2[1]) or ((i1[1]==i2[1]+1 or i1[1]==i2[1]-1) and i1[0]==i2[0])

def DFSiterative(elem, G):
    discovered = []
    S = []
    S.append(elem)
    while S:
        vert = S.pop()
        if not vert in discovered:
            discovered.append(vert)
            adjacent = getAdjacents(vert,G)
            S = S+adjacent
    return discovered

def getAdjacents(vert, G):
    adjacents = []
    i1 = (vert[0]+1,vert[1])
    i2 = (vert[0],vert[1]+1)
    i3 = (vert[0]-1,vert[1])
    i4 = (vert[0],vert[1]-1)
    if checkInGNPos(i1,G):
        adjacents.append(i1)
    if checkInGNPos(i2,G):
        adjacents.append(i2)
    if checkInGNPos(i3,G):
        adjacents.append(i3)
    if checkInGNPos(i4,G):
        adjacents.append(i4)
    return adjacents

def checkInGNPos(index, graph):
    inside = True
    alto, ancho  = graph.shape
    y, x = index
    if y<0 or y>=alto:
        inside = False
    if x<0 or x>=ancho:
        inside = False
    if inside:
        return graph[index] != 0
    else:
        return False

@autojit
def substractRegion(subset, theset):
    return list(set(theset).difference(set(subset)))

def getPredictions(svm):
    '''
    @brief Función que lee las imágenes positivas y negativas, obtiene sus ventanas
    y etiquetas correspondientes y las devuelve
    @return Devuelve dos valores, el primero es la lista de listas de ventanas de
    cada imagen y el segundo es la lista de etiquetas correspondientes para cada imagen
    (no para cada ventana)b
    '''
    pos_imgs = []
    pos_boxes = []
    neg_imgs = []
    # Obtenemos la lista de nombres de imagenes de test positivas
    pos_imgs_names = os.listdir(PATH_TO_INRIA+"/Test/pos")
    for pimg in pos_imgs_names:
        im = cv.imread(PATH_TO_INRIA+"/Test/pos/"+pimg,-1)
        im = np.float32(im)
        # Añadimos las imagenes positivas
        pos_imgs.append(im)
        pos_boxes.append(getPedestrianBoxes(pimg,"/Test/annotations/"))
    del pos_imgs_names
    # Obtenemos la lista de nombres dde imagenes de test negativas
    neg_imgs_names = os.listdir(PATH_TO_INRIA+"/Test/neg")
    for nimg in neg_imgs_names:
        im = cv.imread(PATH_TO_INRIA+"/Test/neg/"+nimg,-1)
        im = np.float32(im)
        # Añadimos las imagenes negativas
        neg_imgs.append(im)
    del neg_imgs_names


    # Obtenemos las respuestas de las imagenes positivas
    print("Obteniendo las predicciones de las imagenes positivas")
    box_pred = getPredPos(pos_imgs[:50],pos_boxes[:50],svm)
    pred_pos = []
    for i in range(len(box_pred)):
        tot = 0
        for pred in box_pred[i]:
            if pred:
                tot+=1
        pred_pos.append(tot/len(pos_boxes[i]))
    # Calculamos las respuestas de las imagenes negativas
    print("Obteniendo las predicciones de las imagenes negativas")
    pred_neg_windows = getPredNeg(neg_imgs[:50],svm)
    pred_neg = []
    for predictions in pred_neg_windows:
        tot = 0
        for pred in predictions:
            if pred==2:
                tot+=1
        pred_neg.append(tot/len(predictions))

    return pred_pos, pred_neg
