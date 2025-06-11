from Lab_Equipment.Config import config

import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
import ctypes
import copy
from IPython.display import display, clear_output

import ipywidgets
import multiprocessing
import time
import scipy.io

from scipy import io, integrate, linalg, signal
from scipy.io import savemat, loadmat
from scipy.fft import fft, fftfreq, fftshift,ifftshift, fft2,ifft2,rfft2,irfft2
# Defult Pploting properties 
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = [2,2]
from typing import List


#SLM Libs
import Lab_Equipment.SLM.pyLCOS as pyLCOS
import Lab_Equipment.ZernikeModule.ZernikeModule as zernMod

#Camera Libs
import Lab_Equipment.Camera.CameraObject as CamForm

# digiHolo Libs
import Lab_Equipment.digHolo.digHolo_pylibs.digholoObject as digholoMod

import Lab_Equipment.digHolo.digHolo_pylibs.digiholoWindowThread as digholoWindowThread


import  Lab_Equipment.AlignmentRoutines.AlignmentFunctions as AlignFunc

import  Lab_Equipment.GeneralLibs.ComplexPlotFunction as cmplxplt

class MeasurmentObj():
    
    def __init__(self,slmObjs: List[pyLCOS.LCOS],
                CamObjs: List[CamForm.GeneralCameraObject],
                digiholoObjs: List[digholoMod.digholoObject]):
        super().__init__()
        self.slmObjs=slmObjs
        self.CamObjs=CamObjs
        # ok so I think it is better to pass a digholo object into the Alignment class as I think it would start to make it very confusing for
        # the user to reference the Alignment object to change the digholo object that is associated with it
        self.digiholoObjs=digiholoObjs 
        
    def GetBatchOfFrames(self,SLMObjIdx=0,pol='H',CamObjIdx=[0],slmChannel=None,modeCount=None,AvgFrameCount=1,modeIdxArr=None):
            CamCount=len(CamObjIdx)
            if slmChannel is None:#if no channel is passed in then use the first active channel on the SLM
                slmChannel=self.slmObjs[SLMObjIdx].ActiveRGBChannels[0]
            if modeCount is None:
                modeCount=self.slmObjs[SLMObjIdx].polProps[slmChannel][pol].modeCount
            modeCount_start=self.slmObjs[SLMObjIdx].polProps[slmChannel][pol].modeCount_start
            modeCount_step=self.slmObjs[SLMObjIdx].polProps[slmChannel][pol].modeCount_step
            for icam in range(CamCount):
                self.CamObjs[CamObjIdx[icam]].SetSingleFrameCapMode()
            # Run a intial batch/AutoAlign of the digholo so see if the systems is starting in a good spot
            if modeIdxArr is not None:
                batchcount=int(len(modeIdxArr))
            else:
                batchcount=int(np.ceil((modeCount-modeCount_start)/modeCount_step))

            Frames=np.empty((CamCount,batchcount*AvgFrameCount,self.CamObjs[CamObjIdx[0]].FrameHeight,self.CamObjs[CamObjIdx[0]].FrameWidth),dtype=np.float32)
            if modeIdxArr is not None:
                iframe=0
                for imode in modeIdxArr:
                    self.slmObjs[SLMObjIdx].setmask(slmChannel,imode)
                    for iavg in range(AvgFrameCount):
                        for icam in range(CamCount): 
                            Frames[icam,iframe,:,:]=self.CamObjs[CamObjIdx[icam]].GetFrame(ConvertToFloat32=True)
                        iframe+=1
            else:
                iframe=0
                for imode in range(modeCount_start,modeCount,modeCount_step):
                    self.slmObjs[SLMObjIdx].setmask(slmChannel,imode)
                    for iavg in range(AvgFrameCount):
                        for icam in range(CamCount): 
                            Frames[icam,iframe,:,:]=self.CamObjs[CamObjIdx[icam]].GetFrame(ConvertToFloat32=True)
                    
                        iframe+=1
            for icam in range(CamCount):   
                self.CamObjs[CamObjIdx[icam]].SetContinousFrameCapMode()
            
            return Frames
    def ProcessBatchOfFrames(self,ObjIdx=0,FramesIn=None,DoAutoAlign=True,
                             AverageBatch=False,batchCount=1,AvgCount=1,
                             maxMG=1,goalIdx=digholoMod.digholoMetrics.IL, 
                             fftWindowSizeY=256,fftWindowSizeX=256,FFTRadius=0.2,
                             digholoThreadObj:digholoWindowThread.digholoWindow=None,plotData=False ):
        self.digiholoObjs[ObjIdx].digholoProperties["FFTRadius"]=FFTRadius
        self.digiholoObjs[ObjIdx].digholoProperties["fftWindowSizeY"]=fftWindowSizeY
        self.digiholoObjs[ObjIdx].digholoProperties["fftWindowSizeX"]=fftWindowSizeX
        # self.digiholoObjs[ObjIdx].digholoProperties["verbosity"]=0
        # self.digiholoObjs[ObjIdx].digholoProperties["resolutionMode"]=0
        self.digiholoObjs[ObjIdx].digholoProperties["maxMG"]=maxMG
        # self.digiholoObjs[ObjIdx].digholoProperties["basisType"]=2

        self.digiholoObjs[ObjIdx].digholoProperties["goalIdx"]=goalIdx
        # self.digiholoObjs[ObjIdx].digholoProperties["goalIdx"]=digholoObj.digholoMetrics.MDL

        # self.digiholoObjs[ObjIdx].digholoProperties["AutoAlignDefocus"]=1
        Frames=copy.deepcopy(FramesIn)

        # Frame=CamXenics.CamObject.GetFrame(True)
        coefs,metrics=self.digiholoObjs[ObjIdx].digHolo_AutoAlign(Frames,
                                                                    AverageBatch=AverageBatch,
                                                                    batchCount=batchCount,
                                                                    AvgCount=AvgCount,
                                                                    DoAutoAlgin=DoAutoAlign)
        CoefsImage,MetricsText=self.digiholoObjs[ObjIdx].GetCoefAndMetricsForOutput()
        
        if (plotData):
            Fullimage,_, WindowSting=self.digiholoObjs[ObjIdx].GetViewport_arr(Frames)
            plt.figure("digholoout")
            plt.subplot(1,2,1)
            plt.imshow(Fullimage)
            plt.subplot(1,2,2)
            # canvasToDispla_Coefs=self.digiholoObjs[ObjIdx].DisplayWindow_GraphWithText(CoefsImage,MetricsText)
            plt.imshow(CoefsImage)

            plt.show()
        print(MetricsText)
        
        if digholoThreadObj is not None:
            #update the digholo thread properties
            digholoThreadObj.Set_digholoWindowProps(self.digiholoObjs[ObjIdx].digholoProperties)
        return coefs,metrics