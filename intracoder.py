import cv2
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras

def loadFaces(impath, newsize=(32, 32), shuffled=False, blured=False, withmirror=False,ychannel=False):
    """
    Loading the faces for training
    :param impath: the path of the directory containing faces
    :param newsize: the desired new size of each face
    :param shuffled: whether to shuffle the faces or not
    :param blured: whether to blur the faces or not
    :param withmirror: whether to add a mirrored face or not
    :param ychannel: whether to transform the color space into YCrCb or not
    :return: an array of faces
    """
    faces=list()
    framedict=json.load(open(impath + '/framedict.json', 'r'))
    for iname in framedict['frames']:

        if ychannel:
            img=cv2.imread(impath+'/'+iname)
            img=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
            img=img[:,:,0]
            img=cv2.resize(img,newsize)/255.0
            faces.append(np.asarray(img).reshape((newsize[0],newsize[1],1)))
            continue

        img=plt.imread(impath + '/' + iname)
        img=cv2.resize(img,newsize)/255.0

        if blured:
            img=cv2.blur(img,(3,3))

        faces.append(img)
        if withmirror:
            faces.append(img[:,::-1,:])

    faces=np.asarray(faces)

    if shuffled:
        np.random.shuffle(faces)

    return faces

def aencoder():
    """
    Constructor for shallow layers of encoder
    :return: shallow layers of encoder
    """
    enc = keras.Sequential()

    enc.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
    enc.add(keras.layers.LeakyReLU())
    enc.add(keras.layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same'))
    enc.add(keras.layers.LeakyReLU())
    enc.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
    enc.add(keras.layers.LeakyReLU())

    return enc

def iencoder():
    """
    Constructor for deeper layers of encoder
    :return: deeper layers of encoder
    """
    enc = keras.Sequential()

    enc.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    enc.add(keras.layers.LeakyReLU())
    enc.add(keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    enc.add(keras.layers.LeakyReLU())
    enc.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    enc.add(keras.layers.LeakyReLU())

    enc.add(keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    enc.add(keras.layers.LeakyReLU())
    enc.add(keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    enc.add(keras.layers.LeakyReLU())
    enc.add(keras.layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same'))
    enc.add(keras.layers.LeakyReLU())

    return enc

def idecoder():
    """
    Constructor for deeper layers of decoder
    :return: deeper layers of decoder
    """
    dec = keras.Sequential()

    dec.add(keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(128, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())

    dec.add(keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())

    return dec

def adecoder():
    """
    Constructor for shallow layers of decoder
    :return: shallow layers of decoder
    """
    dec=keras.Sequential()

    dec.add(keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())

    dec.add(keras.layers.Conv2DTranspose(3, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())

    return dec

def caencoder():
    """
    Constructor for color encoder
    :return: color encoder
    """
    enc = keras.Sequential()

    enc.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', ))
    enc.add(keras.layers.MaxPooling2D((2, 2)))
    enc.add(keras.layers.LeakyReLU())
    enc.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', ))
    enc.add(keras.layers.LeakyReLU())
    enc.add(keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', ))
    enc.add(keras.layers.LeakyReLU())

    enc.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', ))
    enc.add(keras.layers.MaxPooling2D((2, 2)))
    enc.add(keras.layers.LeakyReLU())
    enc.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', ))
    enc.add(keras.layers.LeakyReLU())
    enc.add(keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', ))
    enc.add(keras.layers.LeakyReLU())

    return enc

def cadecoder():
    """
    Constructor for color decoder without the last layer
    :return: color decoder without the last layer
    """
    dec = keras.Sequential()

    dec.add(keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.UpSampling2D((2,2)))
    dec.add(keras.layers.LeakyReLU())

    dec.add(keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.UpSampling2D((2,2)))
    dec.add(keras.layers.LeakyReLU())

    dec.add(keras.layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.UpSampling2D((2, 2)))
    dec.add(keras.layers.LeakyReLU())

    dec.add(keras.layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.UpSampling2D((2, 2)))
    dec.add(keras.layers.LeakyReLU())

    dec.add(keras.layers.Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.UpSampling2D((2, 2)))
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.LeakyReLU())
    dec.add(keras.layers.Conv2DTranspose(8, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.AveragePooling2D((2, 2)))
    dec.add(keras.layers.LeakyReLU())

    return dec

def ccdecoder():
    """
    Constructor for the last layer of color decoder
    :return: last layer of color decoder
    """
    dec = keras.Sequential()

    dec.add(keras.layers.Conv2DTranspose(3, (1, 1), strides=(1, 1), padding='same', use_bias=False))
    dec.add(keras.layers.BatchNormalization())
    dec.add(keras.layers.Activation('sigmoid'))

    return dec

def intraCoderFaceSwap(facepath1='datasets/faces1',
                       facepath2='datasets/faces2',
                       fsize=32,view=False,saverecords=True,logger:logging.Logger=None,
                       **params):
    """
    Conduct face swap by training models
    :param facepath1: path 1 to load first batches of faces (background)
    :param facepath2: path 2 to load second batches of faces (donor)
    :param fsize: the size of the face
    :param view: whether to view the results (only functional when saverecords is True)
    :param saverecords: whether to save the exprecords
    :param logger:a global logger
    :param params: parameters to tune the model
    :return: records and models trained
    """

    epochs1=params.get('epochs1')
    epochs2=params.get('epochs2')
    rounds=params.get('rounds')
    batch_size=params.get('batch_size')
    recordpath=params.get('record_path')
    ticktok=params.get('ticktok')
    verbose=params.get('verbose')

    #if loss is smaller than breakthreshold, the process will be terminated
    breakthreshold=params.get('breakthreshold')

    #if withpretrained is true, the pretrained model will be loaded
    withpretrained=params.get('withpretrained')

    #Construct models
    aenc = aencoder()
    ienc = iencoder()
    idec = idecoder()
    adec1 = adecoder()
    adec2 = adecoder()

    caenc = caencoder()
    cadec = cadecoder()
    ccdec1 = ccdecoder()
    ccdec2 = ccdecoder()

    #Load pretrained weights if designated
    if withpretrained:
        loadWeights([aenc, ienc, idec,adec1,adec2],
                    ['aenc', 'ienc', 'idec','adec','adec'])

        loadWeights([caenc, cadec,ccdec1,ccdec2],
                    ['caenc', 'cadec','ccdec','ccdec'])

    #Construct an optimizer
    optimizer = keras.optimizers.Adam()

    #Build and compile models
    model1 = keras.Sequential([aenc, ienc, idec, adec1])
    model1.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])
    model2 = keras.Sequential([aenc, ienc, idec, adec2])
    model2.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])

    cmodel1 = keras.Sequential([caenc, cadec, ccdec1])
    cmodel1.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])
    cmodel2 = keras.Sequential([caenc, cadec, ccdec2])
    cmodel2.compile(loss='mse', optimizer=optimizer, metrics=['mse','mae'])

    #Set up params and placeholders
    loss1 = []
    loss2 = []
    closs1 = []
    closs2 = []
    ep1 = epochs1 if epochs1 else 1
    ep2 = epochs2 if epochs2 else 1
    rounds = rounds if rounds else 40
    batch_size = batch_size if batch_size else 2
    verbose =verbose if verbose else 1
    breakthreshold=breakthreshold if breakthreshold else 0.001
    es = keras.callbacks.EarlyStopping(monitor='mse', patience=10)

    #Load datasets
    faces1 = loadFaces(facepath1, (fsize, fsize))
    faces2 = loadFaces(facepath2, (fsize, fsize))
    faces1u = loadFaces(facepath1, (fsize*4, fsize*4))
    faces2u = loadFaces(facepath2, (fsize*4, fsize*4))

    t0 = time.time()

    #Training
    for r in range(rounds):

        history1 = model1.fit(faces1, faces1, batch_size=batch_size, epochs=ep1, verbose=verbose, callbacks=[es])
        loss1.extend(history1.history['loss'])
        history2 = model2.fit(faces2, faces2, batch_size=batch_size, epochs=ep2, verbose=verbose, callbacks=[es])
        loss2.extend(history2.history['loss'])

        chistory1 = cmodel1.fit(model1.predict(faces1), faces1u, batch_size=batch_size, epochs=ep1, verbose=verbose, callbacks=[es])
        closs1.extend(chistory1.history['loss'])
        chistory2 = cmodel2.fit(model2.predict(faces2), faces2u, batch_size=batch_size, epochs=ep2, verbose=verbose, callbacks=[es])
        closs2.extend(chistory2.history['loss'])

        if logger:
            logger.info('ITERATION {}'.format(r)+
                        ' CURRENT LOSS: F1 '+str(loss1[-1])+' ;F2 '+str(loss2[-1])+
                        ' ;C1 '+str(closs1[-1])+' ;C2 '+str(closs2[-1]))
        else:
            print('ITERATION {}'.format(r))

        if loss1[-1] <breakthreshold and loss2[-1]<breakthreshold and closs1[-1]<breakthreshold and closs2[-1]<breakthreshold:
            break

    t1 = time.time()

    if ticktok:
        print('Training time used: ', str(t1 - t0))

    #Save the weights
    saveWeights([aenc, ienc, idec, adec2, caenc, cadec, ccdec1],
                ['aenc', 'ienc', 'idec', 'adec', 'caenc', 'cadec', 'ccdec'])

    #Note down an experiment record
    exprecords = dict()
    exppath=recordpath if recordpath else 'expresults'
    exprecords['faceSize'] = fsize
    exprecords['faceShape'] = {'1': faces1.shape, '2': faces2.shape}
    exprecords['epochs'] = {'1': ep1, '2': ep2}
    exprecords['rounds'] = rounds
    exprecords['batchSize'] = batch_size
    exprecords['trainingTime'] = str(t1 - t0)
    exprecords['losses'] = {'s1': loss1, 's2': loss2, 'c1': closs1, 'c2': closs2}

    if saverecords:
        json.dump(exprecords, open(exppath + '/exprecords.json', 'w'), indent=4)
        with open(exppath + '/model.json', 'w') as f:
            f.write(model2.to_json(indent=4))
        with open(exppath + '/cmodel.json', 'w') as f:
            f.write(cmodel1.to_json(indent=4))

        plt.figure()
        plt.title('losses')
        plt.plot(np.log(np.asarray(loss1)), label='s1')
        plt.plot(np.log(np.asarray(loss2)), label='s2')
        plt.plot(np.log(np.asarray(closs1)), '.', label='c1')
        plt.plot(np.log(np.asarray(closs2)), '--', label='c2')
        plt.legend()
        plt.savefig(exppath + '/losses.jpg')

        face1 = faces1[0:1]
        face11 = model2.predict(face1)
        face12 = cmodel1.predict(face11)

        face2 = faces2[0:1]
        face21 = model1.predict(face2)
        face22 = cmodel2.predict(face21)

        plt.figure()
        plt.subplot('231')
        plt.imshow(face1[0])
        plt.axis('off')
        plt.subplot('232')
        plt.imshow(face11[0])
        plt.axis('off')
        plt.subplot('233')
        plt.imshow(face12[0])
        plt.axis('off')
        plt.subplot('234')
        plt.imshow(face2[0])
        plt.axis('off')
        plt.subplot('235')
        plt.imshow(face21[0])
        plt.axis('off')
        plt.subplot('236')
        plt.imshow(face22[0])
        plt.axis('off')
        plt.savefig(exppath + '/decodedfaces.jpg')

        blank = model2.predict(np.zeros((1, fsize, fsize, 3)))
        plt.figure()
        plt.imshow(blank[0])
        plt.axis('off')
        plt.savefig(exppath + '/blankdecoded.jpg')
        if view:
            plt.show()
        plt.close()

    return exprecords,model2,cmodel1

def generateSwaps(inpath='datasets/faces1',
                  middlepath='datasets/faces12-swap',
                  outpath='datasets/faces12',
                  modelpath='models',
                  oprefix='fc', imtype='.jpg',clonemode=0,fsize=32,
                  comparisonpath='datasets/comparison',savecomparison=False,
                  model=None,cmodel=None,logger:logging.Logger=None):
    """
    Generate swapped faces using the trained model
    :param inpath: input path of digested faces
    :param outpath: output path of swapped faces
    :param oprefix: output image prefix
    :param imtype: output image type
    :param fsize: face size used for the training model
    :param comparisonpath: path to save the comparisons
    :param savecomparison: whether to save the comparisons or not
    :param model: pose swap model
    :param cmodel: color correction model
    :param logger: a global logger
    :return: swap record
    """

    finalsize=fsize*2

    #Construct models and load weights if not given
    if model is None:
        aenc = aencoder()
        ienc = iencoder()
        idec = idecoder()
        adec = adecoder()

        loadWeights([aenc, ienc, idec, adec],
                    ['aenc', 'ienc', 'idec', 'adec'],modelpath=modelpath)

        model = keras.Sequential([aenc, ienc, idec, adec])

    if cmodel is None:
        caenc = caencoder()
        cadec = cadecoder()
        ccdec = ccdecoder()

        loadWeights([caenc, cadec, ccdec],
                    ['caenc', 'cadec', 'ccdec'],modelpath=modelpath)

        cmodel = keras.Sequential([caenc,cadec,ccdec])

    #Load datasets for generation
    faces1 = loadFaces(inpath, (fsize, fsize),withmirror=False,shuffled=False)
    framedict=json.load(open(inpath + '/framedict.json', 'r'))

    #Note down a swap record
    swapdict=dict()
    swapdict['faceSize']=fsize
    swapdict['inputPath']=inpath
    swapdict['middlePath']=middlepath
    swapdict['outputPath']=outpath
    swapdict['outputPrefix']=oprefix
    swapdict['outputImtype']=imtype
    swapdict['frames']=list()
    swapdict['skiped']=list()

    t0=time.time()

    #Get the transformed images
    faces10 = model.predict(faces1)
    faces12 = cmodel.predict(faces10)

    nums=faces12.shape[0]
    for i in range(nums):
        inname=framedict['frames'][i]
        outname=oprefix+str(i)+imtype

        f10=faces10[i]
        f12=faces12[i]
        f1=faces1[i]

        fmix=cv2.resize(f12,(finalsize,finalsize))

        plt.imsave(middlepath+'/'+outname,
                   (f10-np.min(f10))/(np.max(f10)-np.min(f10)))
        plt.imsave(outpath + '/' + outname,fmix)

        try:
            #Poisson image blending
            if clonemode==0:
                clone=cv2.NORMAL_CLONE
            else:
                clone=cv2.MIXED_CLONE

            fore = cv2.imread(outpath + '/' + outname)
            back = cv2.resize(cv2.imread(inpath + '/' + inname), (finalsize, finalsize))

            #Construct a mask that cuts the edges
            mask = 255 * np.ones(fore.shape, fore.dtype)
            dedge = 1
            mask[:dedge, :] = 0
            mask[:, :dedge] = 0
            mask[-dedge:, :] = 0
            mask[:, -dedge:] = 0

            width, height, channels = back.shape
            center = (int(height / 2), int(width / 2))
            fmerged = cv2.seamlessClone(fore, back, mask, center, clone)
            cv2.imwrite(outpath + '/' + outname, fmerged)
            fmix = cv2.cvtColor(fmerged, cv2.COLOR_BGR2RGB)

            swapdict['frames'].append(outname)
        except:

            if logger:
                logger.warning('Frame '+str(i)+' is skiped.')

            swapdict['skiped'].append(str(i))
            continue


        if i%20==0 and savecomparison:
            fig=plt.figure()
            plt.subplot('131')
            plt.imshow(f1)
            plt.axis('off')
            plt.subplot('132')
            plt.imshow(f10)
            plt.axis('off')
            plt.subplot('133')
            plt.imshow(fmix)
            plt.axis('off')
            plt.savefig(comparisonpath+'/cp'+str(i)+imtype)
            plt.close(fig)

    t1=time.time()
    swapdict['time']=str(t1-t0)

    json.dump(swapdict,open(outpath+'/framedict.json','w'),indent=4)

    return swapdict

def saveWeights(netlist:list,namelist:list,modelpath:str='models'):
    """
    Save the weights of models in netlist with the corresponding names in namelist into modelpath
    :param netlist: a list models
    :param namelist: a list of names
    :param modelpath: a path for the files
    :return: None
    """
    for net,name in zip(netlist,namelist):
        net.save_weights(modelpath+'/'+name)

def loadWeights(netlist:list,namelist:list,modelpath:str='models'):
    """
    Load the weights of models in netlist with the corresponding names in namelist from modelpath
    :param netlist: a list of models
    :param namelist: a list of names
    :param modelpath: a path for the files
    :return: None
    """
    for net,name in zip(netlist,namelist):
        net.load_weights(modelpath+'/'+name)

def latentSpaceInsight():
    """
    A auxiliary function for the observation of latent spaces
    :return: None
    """
    fsize=32
    aenc = aencoder()
    ienc = iencoder()
    idec = idecoder()
    adec = adecoder()

    caenc=caencoder()
    cadec=cadecoder()
    ccdec=ccdecoder()

    loadWeights([aenc, ienc, idec, adec],
                ['aenc', 'ienc', 'idec', 'adec'])

    loadWeights([caenc, cadec, ccdec],
                ['caenc', 'cadec', 'ccdec'])

    faces1=loadFaces('datasets/faces1',(fsize,fsize))

    od=120

    fts=idec.predict(ienc.predict(aenc.predict(faces1)))
    faces12=adec.predict(fts)

    ftsc=cadec.predict(caenc.predict(faces12))
    faces12c=ccdec.predict(ftsc)

    ftm = fts[od]
    plt.figure()
    for i in range(64):
        f0 = ftm[:, :, i]
        plt.subplot(8, 8, i + 1)
        plt.imshow(f0,cmap='gray')
        plt.axis('off')

    plt.figure()
    for i in range(64):
        ftmi=0*ftm.copy()
        ftmi[:, :, i]=ftm[:,:,i]
        f0 = adec.predict(ftmi.reshape((1,16,16,64)))
        #f0 = (f0-np.min(f0))/(np.max(f0)-np.min(f0))
        #f0 = ccdec.predict(cadec.predict(caenc.predict(f0)))
        plt.subplot(8, 8, i + 1)
        plt.imshow(f0[0],cmap='gray')
        plt.axis('off')

    ftmc = ftsc[od]
    print(ftmc.shape)
    plt.figure()
    for i in range(8):
        f0 = ftmc[:, :, i]
        plt.subplot(2, 4, i + 1)
        plt.imshow(f0,cmap='gray')
        plt.axis('off')

    plt.figure()
    for i in range(8):
        ftmi=0*ftmc.copy()
        ftmi[:, :, i]=ftmc[:,:,i]
        f0 = ccdec.predict(ftmi.reshape((1,128,128,8)))
        #f0 = ccdec.predict(cadec.predict(caenc.predict(f0)))
        plt.subplot(2, 4, i + 1)
        plt.imshow(f0[0])
        plt.axis('off')

    print(fts.shape)
    plt.figure()
    plt.imshow(faces1[od])
    plt.axis('off')
    plt.figure()
    plt.imshow(faces12[od])
    plt.axis('off')
    plt.figure()
    plt.imshow(faces12c[od])
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    #test the function of intraCoderFaceSwap, generateSwaps and latenetSpaceInsight
    exprecords,model,cmodel=intraCoderFaceSwap()
    swapdict=generateSwaps(fsize=exprecords['size'],savecomparison=True,modelpath='models')
    latentSpaceInsight()