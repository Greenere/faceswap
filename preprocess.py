import cv2
import time
import json
import numpy as np
import logging
from mtcnn import MTCNN

def videoRecordAndWrite(path, vname):
    """
    This function records you and save this video.
    :param path: where to save your video
    :param vname: the name of the video
    :return: None
    """
    capture=cv2.VideoCapture(0)

    fourcc=cv2.VideoWriter_fourcc(*'MJPG')
    outfile=cv2.VideoWriter(path+'/'+vname,fourcc,25.,(640,480))

    while capture.isOpened():
        ret,frame=capture.read()
        if ret:
            outfile.write(frame)
            cv2.imshow("frame",frame)
            if cv2.waitKey(1)==ord('q'):
                break

def video2images(videopath, impath, imprefix, start=0, step=1,
                 cutface=True, align=True, withof=False, focused=True,logger:logging.Logger=None):
    """
    This function digests a video into images.
    :param videopath: the path of the video (with video name)
    :param impath: the path of the directory for the output images
    :param imprefix: the prefix of the output images
    :param start: counting start of this round ( this is a pram only used for the function `video2imagesID`, which digests multiple videos)
    :param step: the step of the frame skip, step 1 means each frame is counted
    :param cutface: whether to cut the face region off or not
    :param align: whether to align the face or not (only functional when cutface is True)
    :param withof: whether to add overflow edges to the face (only functional when cutface is True and align is False)
    :param focused: whether to focus the ROI into the region that only contains major lankmarks ( only functional when cutface is True and align is True)
    :param logger: a global logger
    :return: a dictionary containing the infos
    """

    capture=cv2.VideoCapture(videopath)

    frameCount = capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if logger:
        logger.info('Video '+videopath+' has '+str(frameCount)+' frames.')
    else:
        print('Video '+videopath+' has '+str(frameCount)+' frames.')

    framedict=dict()
    framedict['videoPath']=videopath
    framedict['imagePath']=impath
    framedict['imagePrefix']=imprefix
    framedict['settings']={
        'start':start,
        'step':step,
        'cutface':cutface,
        'align':align,
        'withof':withof,
        'focused':focused
    }
    framedict['frameCount']=int(frameCount)
    framedict['frames']=list()
    framedict['skiped']=list()

    mtcnnet=None
    if cutface:
        mtcnnet = MTCNN('./mtcnn.pb')

    for i in range(0,int(frameCount),step):
        capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret,frame=capture.read()

        if cutface:
            if not align:
                box,lm=faceDetection(frame,model=mtcnnet,withof=withof)
                try:
                    face=frame[box[0]:box[2],box[1]:box[3],:]
                except TypeError:
                    if logger:
                        logger.warning('Frame '+str(i)+' is skiped.')
                    framedict['skiped'].append(str(i))
                    continue
            else:
                try:
                    face=faceDetectionAndAlign(frame,model=mtcnnet,focused=focused)
                except:
                    if logger:
                        logger.warning('Frame '+str(i)+' is skiped.')
                    framedict['skiped'].append(str(i))
                    continue
        else:
            face=frame

        if ret:
            cv2.imwrite(impath + '/' + imprefix + str(int(start + i)) + '.jpg', face)
            framedict['frames'].append(imprefix + str(int(start + i)) + '.jpg')
        else:
            if logger:
                logger.warning('Frame ' + str(i) + ' is skiped.')
            framedict['skiped'].append(str(i))

    capture.release()
    json.dump(framedict, open(impath + '/framedict.json', 'w'), indent=4)

    return frameCount

def faceDetection(image,model:MTCNN,withof=True):
    """
    :param image: image on which to detect
    :param model: model with which to detect
    :param withof: whether the output has overflow edges or not
    :return: box (of faces) and lm (landmarks of special points)
    """
    of=10
    bbox, scores, landmarks = model.detect(image)
    try:
        box=bbox[0].astype('int32')
        if withof:
            box[0]-=of
            box[2]+=of
            box[1]-=of
            box[3]+=of
        lm=landmarks[0]
    except:
        box=None
        lm=None
    return box,lm

def faceDetectionAndAlign(image,model:MTCNN,focused=True):
    """
    :param image: image on which to detect
    :param model: model with which to detect
    :param focused: whether to trim the face
    :return: face (the detected face)
    """

    bbox, scores, landmarks = model.detect(image)

    lm=landmarks[0]
    POINTS=np.asarray([[lm[5+i],lm[i]] for i in range(5)],np.float32)

    # the set of REFERENCE_FACIAL_POINTS is quoted from
    # 从零开始搭建人脸识别系统（二）：人脸对齐 https://zhuanlan.zhihu.com/p/61343643
    REFERENCE_FACIAL_POINTS = np.array([
        [30.29459953, 51.69630051],
        [65.53179932, 51.50139999],
        [48.02519989, 71.73660278],
        [33.54930115, 92.3655014],
        [62.72990036, 92.20410156]
    ], np.float32)

    affinematrix=findNonreflectiveSimilarity(POINTS,REFERENCE_FACIAL_POINTS)

    aligned_image=cv2.warpAffine(image.copy(),affinematrix,(100,120))

    if focused:
        face=aligned_image[30:120,15:85,:]
    else:
        face=aligned_image

    return face

def findNonreflectiveSimilarity(uv, xy, K=2):
    """
    This function utilizes least square method to calculate an affine matrix to warp uv to xy
    It is quoted from : 从零开始搭建人脸识别系统（二）：人脸对齐 https://zhuanlan.zhihu.com/p/61343643
    :param uv:  a matrix of points from view 1
    :param xy:  a matrix of points from view 2
    :param K:  a rank threshold
    :return: an affine matrix to warp the image from view 1 to view 2
    """

    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))

    # We know that X * r = U
    if np.linalg.matrix_rank(X) >= 2 * K:
        r, _, _, _ = np.linalg.lstsq(X, U,rcond=None)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])

    T = np.linalg.inv(Tinv)

    T[:, 2] = np.array([0, 0, 1])

    T = T[:, 0:2].T

    return T

def faceDetection_test():
    """
    This function tests whether the face detection model works or not.
    It will open a window powered by cv2 and captures a video stream with your camera.
    :return: None
    """
    capture=cv2.VideoCapture(0)
    model=MTCNN('./mtcnn.pb')
    while True:
        ret,frame=capture.read()
        if not ret:
            break
        box, lm = faceDetection(frame, model=model)
        try:
            faceimg = cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 3)
            for i in range(5):
                faceimg = cv2.circle(faceimg, (lm[i + 5], lm[i]), 1, (0, 255, 0), 2)
        except:
            faceimg=frame
        cv2.imshow('image', faceimg)
        cv2.waitKey(1)

if __name__ == '__main__':
    #test whether it works or not
    t0=time.time()

    video2images(videopath='datasets/id1_0004.mp4',
                 impath='datasets/faces1', imprefix='fa',
                 cutface=True,align=True,withof=False,focused=True)
    video2images(videopath='datasets/id19_0001.mp4',
                 impath='datasets/faces2',imprefix='fb',
                 cutface=True,align=True,withof=False,focused=True)

    t1=time.time()

    print('TIME USED: ',t1-t0)