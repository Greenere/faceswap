import cv2
import os
import time
import imageio
import json
import logging

import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN
from preprocess import faceDetection,findNonreflectiveSimilarity

def images2video(videopath, impath, size=(500, 500)):
    """
    This function turns images into a video
    :param videopath: the path of the output video (with video name)
    :param impath: the path of the directory that contains images and a framedict.json
    :param size: the height and width of the output video
    :return: a dictionary containing infos
    """

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videowriter = cv2.VideoWriter(videopath, fourcc, 25., size)

    framedict=json.load(open(impath+'/framedict.json','r'))

    t0=time.time()

    for framename in framedict['frames']:
        if not os.path.exists(impath+'/'+framename):
            continue
        frame=cv2.imread(impath+'/'+framename)
        frame=cv2.resize(frame,size)
        videowriter.write(frame)

    t1=time.time()

    ivdict=dict()
    ivdict['fourcc']=fourcc
    ivdict['videopath']=videopath
    ivdict['size']=size
    ivdict['time']=str(t1-t0)

    return ivdict

def faceBlending(bgimage, face, model:MTCNN):
    """
    Blending faces without alignment
    :param bgimage: the background
    :param face: the foreground face
    :param model: the detection model
    :return: the blended frame and whether it's successful or not
    """
    box, lm = faceDetection(bgimage, model)
    if box is None:
        return bgimage, False

    bgface = bgimage[box[0]:box[2], box[1]:box[3], :]

    downratex = 0.2
    downratey = 0.2

    fshape=(bgface.shape[1],bgface.shape[0])

    bkernel=(int(fshape[0]/5)*2-1,int(fshape[1]/5)*2-1)
    dedge=5
    emask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(fshape[0],fshape[1]))
    mask=cv2.cvtColor(bgface,cv2.COLOR_BGR2GRAY)
    mask[:dedge,:]=0
    mask[:,:dedge]=0
    mask[-dedge:,:]=0
    mask[:,-dedge:]=0
    mask*=emask
    mask=cv2.GaussianBlur(mask,bkernel,3)
    mask+=255-np.max(mask)
    mask=np.asarray(mask,bgface.dtype)
    ret,mask=cv2.threshold(mask,100,255,cv2.THRESH_OTSU)
    mask = cv2.GaussianBlur(mask, bkernel, 10)

    facimg=cv2.resize(face, fshape)

    box1, lm1 = faceDetection(bgface,model)
    box2, lm2 = faceDetection(facimg, model)

    if box1 is None or box2 is None:
        return bgimage, False

    center =(int((box[1]+box[3])/2*downratey),int((box[0]+box[2])/2*downratex))

    facimg = cv2.resize(facimg, (int(facimg.shape[1] * downratey), int(facimg.shape[0] * downratex)))
    bgimg = cv2.resize(bgimage, (int(bgimage.shape[1] * downratey), int(bgimage.shape[0] * downratex)))
    mask = cv2.resize(mask, (facimg.shape[1], facimg.shape[0]))

    frame2=cv2.seamlessClone(facimg,bgimg,mask,center,cv2.MIXED_CLONE)

    return frame2,True

def faceAlignedBlending(bgimage, face, model:MTCNN,
                        focused=True,view=False,
                        downrate=0.2):
    """
    Blending Aligned Faces (image fusion)
    :param bgimage: the background
    :param face: the foreground face
    :param model: the detection model
    :param focused: whether the face was focused or not
    :param view: whether to display the results or not
    :param downrate: downsampling rate of the input images
    :return: blended image and whether it's successful or not
    """
    box,lm=faceDetection(bgimage, model)
    if box is None:
        return bgimage, False

    downratex=downrate
    downratey=downrate

    REFERENCE_FACIAL_POINTS = np.array([
        [30.29459953, 51.69630051],
        [65.53179932, 51.50139999],
        [48.02519989, 71.73660278],
        [33.54930115, 92.3655014],
        [62.72990036, 92.20410156]
    ], np.float32)

    if focused:
        REFERENCE_FACIAL_POINTS-=np.asarray([15,30],np.float32)

    POINTS = np.asarray([[lm[5 + i], lm[i]] for i in range(5)], np.float32)

    affinematrix = findNonreflectiveSimilarity(REFERENCE_FACIAL_POINTS,POINTS)

    bgshape = (bgimage.shape[1], bgimage.shape[0])
    faceshape = (70,90) if focused else (100,120)

    #reverse affine the aligned face into the original view
    reverseface=cv2.warpAffine(cv2.resize(face, faceshape), affinematrix, bgshape)

    #create a mask for a better border for image fusion
    mask1 = cv2.cvtColor(bgimage.copy(), cv2.COLOR_BGR2YCrCb)
    mask1 = mask1[:,:,0]
    ex=int(bgshape[0]/10*2+1)
    mask1=cv2.dilate(mask1,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ex,ex)))
    mask2=cv2.cvtColor(reverseface.copy(),cv2.COLOR_BGR2GRAY)
    mask2 = cv2.erode(mask2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask2[mask2>0]=255
    mask=cv2.bitwise_and(mask1,mask2)
    mask=mask+255-np.max(mask)
    mask=cv2.bitwise_and(mask,mask2)
    # mask[mask>0]=255

    merged=bgimage/255
    merged[:, :, 0] *= (255 - mask)/255
    merged[:, :, 1] *= (255 - mask)/255
    merged[:, :, 2] *= (255 - mask)/255
    maskedface=reverseface/255
    maskedface[:, :, 0] *= mask / 255
    maskedface[:, :, 1] *= mask / 255
    maskedface[:, :, 2] *= mask / 255

    # examine the blending process
    # plt.figure()
    # plt.subplot('231')
    # plt.imshow(mask1)
    # plt.axis('off')
    # plt.subplot('232')
    # plt.imshow(mask2)
    # plt.axis('off')
    # plt.subplot('233')
    # plt.imshow(mask)
    # plt.axis('off')
    # plt.subplot('234')
    # plt.imshow(maskedface)
    # plt.axis('off')
    # plt.subplot('235')
    # plt.imshow(merged)
    # plt.axis('off')
    # plt.subplot('236')
    # plt.imshow(merged+maskedface)
    # plt.axis('off')
    # plt.show()

    #Down sampling the images

    merged=cv2.resize(merged,(int(merged.shape[1]*downratey),int(merged.shape[0]*downratex)))
    maskedface = cv2.resize(maskedface, (int(maskedface.shape[1] * downratey), int(maskedface.shape[0] * downratex)))

    merged=merged+maskedface

    merged=np.asarray(merged*255,dtype=np.int32)

    if view:
        plt.figure()
        plt.subplot('131')
        plt.title('Reversed Face')
        plt.imshow(cv2.cvtColor(reverseface, cv2.COLOR_BGR2RGB))
        plt.subplot('132')
        plt.title('Mask')
        plt.imshow(mask)
        plt.subplot('133')
        plt.title('Merged Image')
        plt.imshow(cv2.cvtColor(merged, cv2.COLOR_BGR2RGB))
        plt.show()

    return merged,True

def blending(videopath, impath, outpath,
             outprefix='e', imtype='.jpg',downrate=0.2,
             align=True, focused=True,logger:logging.Logger=None):
    """
    Reading a video and blending the frame with faces, then save them into outpath
    :param videopath: the path of video (with video name)
    :param impath: the path of the directory containing faces
    :param outpath: the path of the directory of the output blended images
    :param outprefix: the outprefix of the blended images
    :param imtype: the type of the output image
    :param align: whether the faces were aligned or not
    :param focused: whether the faces were focused or not
    :param logger: a global logger
    :return: a dictionary containing infos about blending
    """
    fsize=None

    capture = cv2.VideoCapture(videopath)

    frameCount = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if logger:
        logger.info(videopath+' has '+str(frameCount)+' frames.')

    blendingdict=dict()
    blendingdict['frameCount']=int(frameCount)
    blendingdict['videoPath']=videopath
    blendingdict['imagePath']=impath
    blendingdict['settings']={
        'outPrefix':outprefix,
        'outImtype':imtype,
        'align':align,
        'focused':focused
    }
    blendingdict['frames']=list()
    blendingdict['skiped']=list()

    swapdict=json.load(open(impath+'/framedict.json','r'))
    skiped=swapdict['skiped']

    mtcnnet = MTCNN('./mtcnn.pb')

    for i in range(int(frameCount)):
        outname= outprefix + str(i) + imtype

        capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, background = capture.read()

        if str(i) not in skiped:
            face = cv2.imread(impath + '/' + swapdict['frames'][i])

            if align:
                blendedframe, success = faceAlignedBlending(background, face, mtcnnet, focused,downrate=downrate)
            else:
                blendedframe, success = faceBlending(background, face, mtcnnet)

            if not success:
                if logger:
                    logger.warning('Frame ' + str(i) + ' is not successfully blended.')
                blendingdict['skiped'].append(str(i))
                continue
        else:
            blendedframe=background

        # if i % 10 == 0:
        #     print('·', end=' ')
        # if i % 200 == 0:
        #     print()

        if ret:
            cv2.imwrite(outpath + '/'+outname, blendedframe)

            blendingdict['frames'].append(outname)
        else:

            if logger:
                logger.warning('Frame '+str(i)+' is not successfully blended.')

            blendingdict['skiped'].append(str(i))
            continue

        fsize=(blendedframe.shape[1],blendedframe.shape[0])

    capture.release()

    blendingdict['fsize']=fsize
    json.dump(blendingdict,open(outpath+'/framedict.json','w'),indent=4)

    return blendingdict

def create_gif(gifname='datasets/fswap.gif',impath='datasets/output12',duration=1/30):
    """
    Create a gif using images from impath with a duration of duration
    :param gifname: the path of the output gif (with name)
    :param impath: the path of the directory containing images
    :param duration: the duration of each frame
    :return: None
    """

    framedict = json.load(open(impath + '/framedict.json', 'r'))

    frames=list()
    for framename in framedict['frames']:
        frames.append(imageio.imread(impath+'/'+framename))

    imageio.mimsave(gifname, frames, 'GIF', duration=duration)

if __name__ == '__main__':
    #test the function of blending and images2video
    blendingdict=blending(videopath='datasets/id1_0004.mp4',
                          impath='datasets/faces12',
                          outpath='datasets/output12',downrate=0.5)
    ivdict=images2video(videopath='datasets/generated/id1-id19-4-1.avi',
                        impath='datasets/output12',
                        size=blendingdict['fsize'])
    create_gif(gifname='datasets/generated/id1-id19-4-1.gif')