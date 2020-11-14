import time
import logging
import logging.handlers

from preprocess import video2images
from intracoder import intraCoderFaceSwap,generateSwaps
from postprocess import blending,images2video,create_gif

def getLogger(filename='swap.log'):
    """
    Construct a global logger
    :param filename: the name of the log file
    :return: a global logger
    """
    logger = logging.Logger(filename.split('.')[0])
    logger.setLevel(level=logging.INFO)
    maxsize=1*1024*1024
    handler=logging.handlers.RotatingFileHandler(filename=filename,encoding='utf-8',maxBytes=maxsize)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def faceSwapPipeLine(video1,video2,video12path,fsize=32,downrate=0.5):
    """
    An end-to-end pipeline to generate deepfakes
    :param video1: the original video (the background video)
    :param video2: the target video (the foreground video, the face in which will be swapped into the background video)
    :param video12path: the save path (with name) of the generated video
    :param fsize: the size of the training faces in intracoder module
    :param downrate: the down sampling rate of the final video (compared to the original video)
    :return: None
    """

    logger=getLogger('logs/swap.log')

    logger.info('Digesting videos···')

    t0=time.time()

    #digesting videos into images frame by frame
    video2images(videopath=video1,
                 impath='datasets/faces1', imprefix='fa',
                 cutface=True, align=True, withof=False, focused=True,logger=logger)
    video2images(videopath=video2,
                 impath='datasets/faces2', imprefix='fb',
                 cutface=True, align=True, withof=False, focused=True,logger=logger)

    t1=time.time()

    logger.info('Videos digested, time used: '+str(t1-t0)+'s')
    logger.info('FaceSwap model training···')

    #training face swap models with images retreived
    #model is used for face swap
    #colormodel is used for color correction
    exprecords, model, colormodel = intraCoderFaceSwap(facepath1='datasets/faces1',
                                                   facepath2='datasets/faces2',
                                                   fsize=fsize,logger=logger,
                                                   breakthreshold=0.0005,
                                                   withpretrained=True,
                                                   epochs1=1,
                                                   epochs2=1,
                                                   rounds=2,
                                                   batch_size=2,
                                                   saverecords=False)

    #generate face swapped images
    generateSwaps(inpath='datasets/faces1',
                  outpath='datasets/faces12',
                  fsize=exprecords['faceSize'],savecomparison=True,
                  model=model, cmodel=colormodel,logger=logger)

    t2=time.time()

    logger.info('FaceSwap model trained, time used: '+str(t2-t1)+'s')
    logger.info('Blending···')

    #blend swapped faces into images of background
    blendingdict = blending(videopath=video1,
                            impath='datasets/faces12',
                            outpath='datasets/output12',
                            downrate=downrate,focused=True,
                            logger=logger)

    t3=time.time()
    logger.info('Face blended, time used: '+str(t3-t2)+'s')
    logger.info('Constructing faceswap video···')

    #compose frames of images into video
    images2video(videopath=video12path,
                 impath='datasets/output12',
                 size=blendingdict['fsize'])

    #compose frames of images into gif
    create_gif(gifname=video12path.split('.')[0]+'.gif',
               impath='datasets/output12',
               duration=1/30)

    t4=time.time()
    logger.info('Faceswap video constructed, time used: '+str(t4-t3))
    logger.info('The whole generation process is finished'
                ', you may find the video in '+video12path)
    logger.info('Total time used: '+str(t4-t0)+'s')

if __name__ == '__main__':
    faceSwapPipeLine('datasets/id1_0004.mp4',
                     'datasets/id19_0001.mp4',
                     'datasets/generated/id1-id19-4-1.avi',
                     fsize=32)