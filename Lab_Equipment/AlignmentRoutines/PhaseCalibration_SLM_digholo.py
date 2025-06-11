from Lab_Equipment.Config import config 

# Python Libs
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
plt.rcParams['figure.figsize'] = [5,5]

#SLM Libs
import Lab_Equipment.SLM.pyLCOS as pyLCOS
import Lab_Equipment.ZernikeModule.ZernikeModule as zernMod

#Camera Libs
import Lab_Equipment.Camera.CameraObject as CamForm


#Camera Libs
import Lab_Equipment.Camera.CameraObject as CamForm

# digiHolo Libs
import Lab_Equipment.digHolo.digHolo_pylibs.digholoObject as digholoLib


import  Lab_Equipment.GeneralLibs.ComplexPlotFunction as cmplxplt


def ProcessFramesFromPhaseCal(FrameBuffer,digholoObj:digholoLib.digholoObject, MaskNum,FFTRadiusIn=0.2,wavelength=1550e-9,Nx=256,Ny=256,CampixelSize=30e-6):
    digholoObj.digholoProperties["FFTRadius"]=FFTRadiusIn
    digholoObj.digholoProperties["fftWindowSizeX"]=Nx
    digholoObj.digholoProperties["fftWindowSizeY"]=Ny
    digholoObj.digholoProperties["wavelenght"]=wavelength



    Frame_Initial= copy.deepcopy(FrameBuffer[-1,:,:])
    digholoObj.digHolo_AutoAlign(Frame_Initial)
    #Display he initial frame
    Fullimage ,ViewPortRGB_cam,WindowString=digholoObj.GetViewport_arr(Frame_Initial)
    plt.figure()
    plt.imshow(Fullimage)
    plt.show()
    
    digholoObj.digHolo_ProcessBatch(FrameBuffer[0:-1,:,:])
    Fields=digholoObj.digHolo_GetFields()
    
    NewFileForBatch="phaseCal_" + str(int(wavelength*1e9)) + "MaskNum"+str(MaskNum)
    digholoObj.SaveBatchFile(NewFileForBatch,FrameBuffer[0:-1,:,:],True)

    return Fields,WindowString
    

#NOTE when you do a phase calibration you want to start at the 0 gray scale and move up through it. 
# If you dont do this the calibration has a lot of trouble when it flick around from 255 to 0 grey level
# Physically it really shouldnt matter but the SLM really hate going from 255 to 0 so makes a little bit of
# scence from that perspective. This took a week of my time as the phase cals where just absoultely terrible 
# that where coming out.
# Daniel 10min from writting this comment:
# Past Daniel is a absoulte idiot if you think about it for like 10 seconds you were doing
# the phase cal wrong. you have to start it off at 0 grey level and move it up to 255 as this is the
# whole point of the calibration. you are a idiot. I am leaving the comment here so you can feel 
# the shame every time you look at this code.
def PhaseCalibration(slm:pyLCOS.LCOS,channel,CamObj:CamForm.GeneralCameraObject,Direction="x", imask=0,pol="V"):
    
    CamObj.SetSingleFrameCapMode()
    phaseLevels=256
    masksize=slm.polProps[channel][pol].masksize
    
    Nx=masksize[0]
    Ny=masksize[1]
    
    y_center = slm.AllMaskProperties[channel][pol][imask].center[0]
    x_center = slm.AllMaskProperties[channel][pol][imask].center[1]   
    
    FrameBuffer = np.zeros((phaseLevels+1, CamObj.FrameHeight, CamObj.FrameWidth), dtype=np.float32)

    MASK=np.zeros((Nx,Ny),dtype=np.uint8)
    for level in range(phaseLevels):
        print(level, end=' ')
        # Create phase wrap 
        # MASK[:,0:int((Nx/2))]=128
        # MASK[:,int((Nx/2)):Nx]=level
        # MASK[0:int((Ny/2)),:]=128
        # MASK[int((Ny/2)):Ny,:]=level
        if(Direction=="y"):
            MASK[0:int((Ny/2)),:]=128
            MASK[int((Ny/2)):Ny,:]=level
        elif(Direction=="x"):
            MASK[:,0:int((Nx/2))]=128
            MASK[:,int((Nx/2)):Nx]=level
            
        MASKTODisplay_256=slm.Draw_Single_Mask( x_center, y_center, MASK,255)

        slm.Write_To_Display(MASKTODisplay_256,channel)
        
        FrameBuffer[level,:,:]=CamObj.GetFrame(True)
        
    slm.LCOS_Clean(channel)
    
    FrameBuffer[-1,:,:]=CamObj.GetFrame(True)
    
    #Turn continous mode back on for the camera
    CamObj.SetContinousFrameCapMode(CamObj.Exposure)


    return FrameBuffer
