import numpy as np
import cv2 as cv
import auxFunc as af
import pdb
import random

################################################################################
##                        1: Normalización Gamma                              ##
################################################################################

def gammaNormalization(img,c1=1,c2=0.5):
    '''
    @brief Función que corrige el valor gamma usando la ley de potencias con
    coeficientes c1 y c2.
    @param img Imagen a la que queremos normalizarle el valor gamma.
    @param c1 Factor multiplicativo en la normalización, por defecto 1.
    @param c2 Exponente de la normalización, por defecto 1/2 (compressing normalization).
    '''
    return ((np.power((img/255.0)*c1,c2))*255).astype(np.float32)

################################################################################
##                        2: Cómputo del gradiente                            ##
################################################################################

def gradientComputation1DPaper(img,sigma=0):
    '''
    @brief Funcion que computa el gradiente utilizando la mascara 1D [-1,0,1]
    @param img Imagen sobre la que calcular el gradiente
    @param sigma varianza del nucleo gaussiano con el que se convoluciona la imagen
    @return Devuelve una matriz que contiene el gradiente computado sobre la imagen
    '''
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,(0,0),sigma)
    # Calculamos df/dx para todos los canales (RGB)
    outputSignalsdx = af.convoluteWith1DMask([-1,0,1],True,imgAux)
    # Calculamos df/dy para todos los canales (RGB)
    outputSignalsdy = af.convoluteWith1DMask([-1,0,1],False,imgAux)
    # En cada pixel el gradiente es el gradiente del canal con mayor norma (RGB)
    return af.getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt1(img,sigma=0):
    '''
    @brief Funcion que computa el gradiente usando la mascara [-1,1]
    @param img Imagen sobre la que calcular el gradiente
    @param sigma varianza del nucleo gaussiano con el que se convoluciona la imagen
    @return Devuelve una matriz que contiene el gradiente computado sobre la imagen
    '''
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,(0,0),sigma)
    outputSignalsdx = af.convoluteWith1DMask([-1,1],True,imgAux)
    outputSignalsdy = af.convoluteWith1DMask([-1,1],False,imgAux)
    return af.getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt2(img,sigma=0):
    '''
    @brief Funcion que computa el gradiente usando la mascara [1,-8,0,8,-1]
    @param img Imagen sobre la que calcular el gradiente
    @param sigma varianza del nucleo gaussiano con el que se convoluciona la imagen
    @return Devuelve una matriz que contiene el gradiente computado sobre la imagen
    '''
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,(0,0),sigma)
    outputSignalsdx = af.convoluteWith1DMask([1,-8,0,8,-1],True,imgAux)
    outputSignalsdy = af.convoluteWith1DMask([1,-8,0,8,-1],False,imgAux)
    return af.getGradient(outputSignalsdx, outputSignalsdy)

def gradientComputation1DAlt3(img,sigma=0):
    '''
    @brief Funcion que computa el gradiente usando una mascara 2D
    @param img Imagen sobre la que calcular el gradiente
    @param sigma varianza del nucleo gaussiano con el que se convoluciona la imagen
    @return Devuelve una matriz que contiene el gradiente computado sobre la imagen
    '''
    imgAux = None
    if sigma==0:
        imgAux = np.copy(img)
    else:
        imgAux = cv.GaussianBlur(img,(0,0),sigma)
    mask1 = np.array([[0,-1],[1,0]])
    outputSignalsdx = cv.filter2D(img,-1,mask1)
    mask2 = np.array([[-1,0],[0,1]])
    outputSignalsdy = cv.filter2D(img,-1,mask2)

    return af.getGradient(outputSignalsdx, outputSignalsdy)

################################################################################
##                      3: Spatial/Orientation Binning                        ##
################################################################################

def spatialOrientationBinning(dx, dy, tam_cel=3, num_cols=9):
    '''
    @brief Función que dada una matriz de gradientes y un tamaño de célula divide la matriz en
    células, calcula los histogramas de todas y los devuelve en un vector.
    @param gradients Matriz con los gradientes
    @param tam_cel Tamaño de la célula, por defecto 3.
    @param num_cols Numero de columnas del histograma, por defecto 9.
    '''
    # Obtiene el número de filas y columnas de la imagen
    rows, cols = dx.shape
    mag, angle = cv.cartToPolar(dx, dy, angleInDegrees=True)

    # Inicializa los histogramas
    histograms = []

    # Divide la matriz en celdas y llama con cada una al cálculo de histogramas.
    for i in range(0,rows,tam_cel):
        row_histograms = []
        for j in range(0,cols,tam_cel):
            if i+tam_cel<rows and j+tam_cel<cols:
                subMag = mag[i:i+tam_cel,j:j+tam_cel]
                subAng = angle[i:i+tam_cel,j:j+tam_cel]
                hist = af.computeHistogram(subMag,subAng,num_cols)
                row_histograms.append(hist)
        if len(row_histograms)>0:
            histograms.append(row_histograms)
    return np.array(histograms).reshape((int(rows/tam_cel),int(cols/tam_cel),num_cols))

################################################################################
##                 4: Normalization and Descriptor Blocks                     ##
################################################################################

def normalizeDescriptor(bloque):
    '''
    @brief Función que dada una submatriz la normaliza
    @param bloque Submatriz que queremos normalizar
    @return Pasamos el bloque a una lista 1-D y lo dividimos por su norma
    '''
    ret = bloque.reshape(-1)
    norma = np.linalg.norm(ret)
    value = list(ret/norma) if norma!=0 else list(np.zeros(ret.shape[0]))
    return value

def rhog(histogramas,tam_bloque=(2,2)):
    '''
    @brief Función que calcula los descriptores normalizados a partir de los
    histogramas de cada celula dentro de un mismo bloque
    @param histogramas Todos los histogramas computados a partir de celulas
    @param tam_bloque Tamano del bloque debe ser una pareja e.g. (2,2)
    @return Devuelve un array que separa en bloques los histogramas
    '''
    n, m, k = histogramas.shape
    descriptores = []
    for i in range(n-tam_bloque[0]):
        descriptoresFila = []
        for j in range(m-tam_bloque[1]):
            descriptor = normalizeDescriptor(histogramas[i:i+tam_bloque[0],j:j+tam_bloque[1]])
            descriptoresFila.append(descriptor)
        descriptores.append(descriptoresFila)
    return np.array(descriptores)

################################################################################
##                           5: Classification                                ##
################################################################################


def trainSVM(trainData):
    '''
    @brief Función que crea una SVM y la entrena con los datos trainData
    @param trainData Datos con los que queremos entrenar la SVM
    @return Devuelve una SVM ya entrenada
    '''
    svm = cv.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    svm.setTermCriteria((cv.TERM_CRITERIA_COUNT or cv.TERM_CRITERIA_EPS, int(1e7), 1e-6));
    svm.setGamma(0)
    svm.setKernel(cv.ml.SVM_LINEAR);
    svm.setNu(0.5)
    svm.setP(0.1)
    svm.setType(cv.ml.SVM_C_SVC)
    # Ponemos una soft SVM como especifica en el paper
    svm.setC(0.01)
    svm.train(trainData)
    return svm

def testSVM(svm, testData):
    '''
    @brief Función que dada una SVM ya entrenada y unos datos de prueba devuelve lo predicho
    por la SVM para dichos datos
    @param svm SVM con la que queremos realizar la predicción
    @param testData Datos sobre los que queremos predecir
    '''
    retval, results = svm.predict(testData)
    return results

def obtainDescriptors(imgs,silent=False):
    '''
    @brief Función que dada un vector de imágenes obtiene los descriptores asociados
    a la misma y hace la unión. Las imágenes tienen que ser parches de 64x128
    @param imgs Lista de imágenes sobre las que queremos extraer los descriptores
    @return Devuelve un numpy array de descriptores, uno por imagen
    '''
    #---- Print de información ----#
    if not silent:
        print("Normalización Gamma")
    contador=1
    #---- Fin del print de información ----#

    gamma_corrected = []
    for im in imgs:

        #---- Print de información ----#
        if not silent and (contador%100==0 or contador==len(imgs) or contador==1):
            print("Normalizando " + str(contador) + "/" + str(len(imgs)))
        contador+=1
        #---- Fin del print de información ----#

        # Aplicamos la normalización gamma a cada imagen
        gamma_corrected.append(gammaNormalization(im))

    #---- Print de información ----#
    if not silent:
        print("Calculando los gradientes")
    contador=1
    #---- Fin del print de información ----#

    gradients = []
    for gam in gamma_corrected:

        #---- Print de información ----#
        if not silent and (contador%100==0 or contador==len(imgs) or contador==1):
            print("Calculando los gradientes " + str(contador) + "/" + str(len(imgs)))
        contador+=1
        #---- Fin del print de información ----#

        # Calculamos los gradientes de cada imagen
        gradients.append(gradientComputation1DPaper(gam))
    del gamma_corrected

    #---- Print de información ----#
    if not silent:
        print("Calculando los histogramas")
    contador=1
    #---- Fin del print de información ----#

    histograms = []
    for gra in gradients:

        #---- Print de información ----#
        if not silent and (contador%100==0 or contador==len(imgs) or contador==1):
            print("Calculando los histogramas " + str(contador) + "/" + str(len(imgs)))
        contador+=1
        #---- Fin del print de información ----#

        # Calculamos los histogramas de cada matriz de gradientes
        histograms.append(spatialOrientationBinning(gra[0],gra[1]))
    del gradients

    #---- Print de información ----#
    if not silent:
        print("Calculando los descriptores de imagen")
    contador=1
    #---- Fin del print de información ----#

    # Normalizamos por bloques
    img_descr = rhog(histograms[0]).reshape(-1).astype(np.float32)
    for histo in histograms[1:]:

        #---- Print de información ----#
        if not silent and (contador%100==0 or contador==len(imgs) or contador==1):
            print("Calculando el descriptor final " + str(contador) + "/" + str(len(imgs)))
        contador+=1
        #---- Fin del print de información ----#

        # Unimos los descriptores en una sola lista
        descr = rhog(histo).reshape(-1).astype(np.float32)
        img_descr = np.vstack([img_descr,descr])
    del histograms
    return np.array(img_descr)

def obtainTrainData():
    '''
    @brief Función que obtiene todos los datos de entrenamiento cargando las imágenes
    correspondientes y devuelve un objeto de tipo TrainData para SVM
    @return Objeto de tipo TrainData para entrenar la SVM
    '''
    # Cargamos las imágenes de entrenamiento
    img_pos,img_neg = af.loadTrainImgs()

    # Generamos las respuestas 1 si es una persona, 2 si no lo es
    resp = [1 for i in range(len(img_pos))]+[2 for i in range(len(img_neg))]
    resp = np.array(resp)
    # Obtenemos los descriptores, uno por imagen
    img_descr = obtainDescriptors(img_pos + img_neg)
    del img_pos
    del img_neg
    # Creamos los datos de entrenamiento y los devolvemos
    return cv.ml.TrainData_create(img_descr,cv.ml.ROW_SAMPLE,resp.astype(np.int))

def createTestData(chunk_size=100):
    # Este codigo es para usar los croped test
    '''
    imgs_pos,imgs_neg = af.loadTestImgs()
    tags_pos = [1 for i in range(len(imgs_pos))]
    tags_neg = [2 for i in range(len(imgs_neg))]
    imgs = imgs_pos+imgs_neg
    tags = tags_pos+tags_neg
    img_descr = obtainDescriptors( imgs )
    return img_descr, tags
    '''
    # Cargo las imagenes y las tags
    imgs, trueTags = af.getImagesAndTags( )

    # Las pongo en una lista
    imgs_lista = []
    for lim in imgs:
        for im in lim:
            imgs_lista.append(im)

    img_descr = obtainDescriptors(imgs_lista)
    return img_descr, trueTags

def chunkPredictions(vim, vpred):
    '''
    @brief Función que devuelve el vector de predicciones corregidas y obtenidas
    comprobando por ventanas
    @param vim Lista de listas de ventanas
    @param vpred Vector de predicciones para cada ventana
    @return Devuelve una predicción para cada imagen y no para cada ventana
    '''
    ret = []
    pos_vpred=0
    # Para cada lista de ventanas
    for i in range(len(vim)):
        encontradoUnaPersona = False
        # Para cada ventana en la lista de ventanas
        for j in range(len(vim[i])):
            # Con que una sola de las etiquetas predichas para una ventana de una imagen
            # sea un 1 (hay peaton) entonces decimos que la respuesta general para la imagen
            # es 1, hay peatón en la misma
            if vpred[pos_vpred] == 1:
                encontradoUnaPersona = True
            pos_vpred+=1
        # Si hemos encontrado una persona añadimos un 1 y si no un 2
        if encontradoUnaPersona:
            ret.append(1)
        else:
            ret.append(2)
    return ret
