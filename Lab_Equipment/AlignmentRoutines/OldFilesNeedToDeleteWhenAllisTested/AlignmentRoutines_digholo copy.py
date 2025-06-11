from Lab_Equipment.Config import config

# import tomography.standard as standard
# import tomography.masks as masks

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
# import PyCapture2 # module for the camera
# if (config.PYCAPTURE_IMPORT):
#     import PyCapture2
# else:
#     from vmbpy import *
from scipy import io, integrate, linalg, signal
from scipy.io import savemat, loadmat
from scipy.fft import fft, fftfreq, fftshift,ifftshift, fft2,ifft2,rfft2,irfft2
# Defult Pploting properties 
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = [5,5]

# from script_functions import start_worker
# import CameraWindowForm as CamForm
#SLM Libs
import Lab_Equipment.SLM.pyLCOS as pyLCOS
import Lab_Equipment.ZernikeModule.ZernikeModule as zernMod

#Camera Libs
import Lab_Equipment.Camera.CameraObject as CamForm

# digiHolo Libs
import Experiments.Lab_Equipment.digHolo.digHolo_pylibs.digiholoHeader_old as digH_hpy  # as in header file for python... pretty clever I know (Daniel 2 seconds after writing this commment. Head slap you are a idiot )
import Lab_Equipment.digHolo.digHolo_pylibs.digiholoWindowForm as digForm
import Lab_Equipment.digHolo.digHolo_pylibs.digholoCombinedFunction as digholoFuncWrapper

import  Lab_Equipment.AlignmentRoutines.AlignmentFunctions as AlignFunc

# Daniel's Python Libs
import  Lab_Equipment.MyPythonLibs.OpticalOperators as OpticOp
import  Lab_Equipment.MyPythonLibs.ComplexPlotFunction as cmplxplt
import  Lab_Equipment.MyPythonLibs.GaussianBeamBasis as GaussBeams
import  Lab_Equipment.MyPythonLibs.GeneralFunctions as GenFuncs
import  Lab_Equipment.MyPythonLibs.AnalysisFuncitons as ProCouplMat
import  Lab_Equipment.MyPythonLibs.CoupMatrixAndMetricAnalysisFuncitons as MetricCals
import  Lab_Equipment.MyPythonLibs.SaveMaskToBinFile as SaveMaskBin
import  Lab_Equipment.MyPythonLibs.ModelabProcessBatchFile as Modelab

# Ok so i dont know if this will work but I am trying to make a generic golden search function that i can
# pass a function too that is described in a class.
# I can confirm that it worked. Essentiall you make class that has the function in it and as lone as the function only has on input and one output it will work 

class CenterAlginmentObj_digiholo():
    def __init__(self,slmObject:pyLCOS.LCOS,slmChannel ,CamObj:CamForm.GeneralCameraObject,DescreteSpaceArrayX,DescreteSpaceArrayY):
        super().__init__()
        self.slm=slmObject
        self.CamObj=CamObj
        self.PiFlip_cmplx =np.ones((self.slm.LCOSsize[0],self.slm.LCOSsize[1]),dtype=complex)
        # PiFlip_cmplx[:,:]=np.exp(1j*np.pi)
        self.flipdir=0
        self.Phasedir=0
        self.channel=slmChannel
        self.imask=0
        self.pol='V'
        self.ApplyZernike=True
        self.DescreteSpaceArrayX=DescreteSpaceArrayX
        self.DescreteSpaceArrayY=DescreteSpaceArrayY
        
        self.AvgFrameCount=10
        
        
        self.CamObj.SetSingleFrameCapMode()
        PixelSize=self.CamObj.PixelSize
        #Centre wavelength (nanometres)
        lambda0 = 1550e-9
        #Polarisation components per frame
        polCount = 1 
        #Width/height of window to FFT on the camera. (pixels)
        CamDims=CamObj.FrameBuffer.shape
        frameWidth = CamDims[1]
        frameHeight = CamDims[0]
        nx = 320
        ny = 320
        #Amount of detail to print to console. 0: Console off. 1: Basic info. 2:Debug mode. 3: You've got serious issues
        verbosity = 0
        #Sets the resolution mode of the reconstructed field.
        #0 : Full resolution. Reconstructed field will have same pixel
        #size/dimensions as the FFT window.
        #1 : Low resolution. Reconstructed field will have dimension of the IFFT
        #window. 
        resolutionMode = 0
        consoleRedirectToFile=True
        maxMG=1
        AutoAlginFlagsCount=5
        AutoAlginFlags = np.zeros(AutoAlginFlagsCount,dtype=int)
        for i in range(AutoAlginFlagsCount):
            if (i==AutoAlginFlagsCount-1):
                AutoAlginFlags[i]=0
            else:
                AutoAlginFlags[i]=1
        fftWindowSizeX=256
        fftWindowSizeY=256
        self.handleIdx=digH_hpy.digHolo.digHoloCreate()

        #Clear the SLM
        # self.slm.LCOS_Clean()
        # need to put a blank mask with zernikes
        Nx=self.slm.masksize[0]
        Ny=self.slm.masksize[1]
        MASK=np.zeros((Nx,Ny),dtype=complex)
        x_center_Input=int(self.slm.AllMaskProperties[self.channel][self.pol][self.imask].center[1])
        y_center_Input=int(self.slm.AllMaskProperties[self.channel][self.pol][self.imask].center[0])
        if (self.ApplyZernike):
                    MASK_PlussZernike=self.slm.ApplyZernikesToMask(self.channel,np.angle(MASK),imask=0,pol=self.pol,ipol=1,imode=0)
        else:
            MASK_PlussZernike=np.angle(MASK)
        MASKTODisplay=self.slm.Draw_Single_Mask( x_center_Input,y_center_Input, MASK_PlussZernike)
            
        MASKTODisplay_256 = self.slm.phaseTolevel(MASKTODisplay)# Note that the -1*np.pi is so that the background is set to black it really doesn't matter though.
        # Display on SLM
        # slm.LCOS_Display(slm.LCOS_Screen_temp.astype(int), ch = 0)
        self.slm.LCOS_Display(MASKTODisplay_256, self.channel)


        #Get a frame for to make the reference field
        self.CamObj.GetFrame()
        FrameBufferInitial=self.CamObj.FrameBuffer.astype(np.float32)
        frameBufferPtr_initial = FrameBufferInitial.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        batchCount=1
        FFTRadius=0.4
        digH_hpy.digHolo.digHoloConfigSetFrameDimensions(self.handleIdx,frameWidth,frameHeight)
        digH_hpy.digHolo.digHoloConfigSetFramePixelSize(self.handleIdx,PixelSize)
        digH_hpy.digHolo.digHoloConfigSetBatchCount(self.handleIdx,batchCount) 
        digholoFuncWrapper.digholoSetProps(self.handleIdx,polCount,fftWindowSizeX,fftWindowSizeY,FFTRadius,lambda0,resolutionMode,maxMG,AutoAlginFlags)
        digholoFuncWrapper.digholo_AutoAlginBatch(self.handleIdx,batchCount,polCount,
                                                        fftWindowSizeX,fftWindowSizeY,FFTRadius,lambda0,maxMG,resolutionMode,verbosity,
                                                        AutoAlginFlags,frameBufferPtr_initial)
        digholoFuncWrapper.ProcessBatchOfFrames(self.handleIdx,batchCount,FrameBufferInitial)
        #This is the reference Field that the other fields will be overlaped with
        self.Field_Ref=(digholoFuncWrapper.GetField(self.handleIdx))
        self.RefPWR=np.sum(np.abs(OpticOp.overlap(self.Field_Ref,np.conj(self.Field_Ref)))**2)
        plt.imshow(cmplxplt.ComplexArrayToRgb(np.squeeze(self.Field_Ref)));
        def __del__(self):
            ErrorCode=digH_hpy.digHolo.digHoloDestroy(self.handleIdx) 
            print(ErrorCode)

        self.CamObj.SetContinousFrameCapMode(self.CamObj.Exposure.value)
        
        
        
    def PerformCenterAlignment(self):#,TotalSpaceArrX,TotalSpaceArrY):
        self.CamObj.SetSingleFrameCapMode()
        self.flipdir=0 # flipdir X
        #Need to do 2 flips in the same direction for a better center one has the flip reversed
        self.Phasedir=0
        minValX_1,minIdxX_1=AlignFunc.GoldenSelectionSearch(self.DescreteSpaceArrayX[0],self.DescreteSpaceArrayX[-1],1,self.ChangePiFlipTake_digiholo)
        
        MinXCenter_1=minIdxX_1#self.DescreteSpaceArrayX[int(minIdxX_1)]
        # reversed flip in same direction
        self.Phasedir=1
        minValX_2,minIdxX_2=AlignFunc.GoldenSelectionSearch(self.DescreteSpaceArrayX[0],self.DescreteSpaceArrayX[-1],1,self.ChangePiFlipTake_digiholo)
        # MinXCenter_2=self.DescreteSpaceArrayX[int(minIdxX_2)]
        MinXCenter_2=minIdxX_2#self.DescreteSpaceArrayX[int(minIdxX_2)]
        
        # Take the average of the 2 centers
        MinXCenter=(MinXCenter_1+MinXCenter_2)/2.0

        print("min x center values")
        print(MinXCenter_1,MinXCenter_2)
        
        #Now we will go in the the other direction, again you need to do 2 flips for a better results 
        #Need to do 2 flips in the same direction for a better center one has the flip reversed
        self.flipdir=1
        self.Phasedir=0
        minValY_1,minIdxY_1=AlignFunc.GoldenSelectionSearch(self.DescreteSpaceArrayY[0],self.DescreteSpaceArrayY[-1],1,self.ChangePiFlipTake_digiholo)
        # MinYCenter_1=self.DescreteSpaceArrayY[int(minIdxY_1)]
        MinYCenter_1=minIdxY_1
        
        # reversed flip in same direction
        self.Phasedir=1
        minValY_2,minIdxY_2=AlignFunc.GoldenSelectionSearch(self.DescreteSpaceArrayY[0],self.DescreteSpaceArrayY[-1],1,self.ChangePiFlipTake_digiholo)
        # MinYCenter_2=self.DescreteSpaceArrayY[int(minIdxY_2)]
        MinYCenter_2=minIdxY_2
        
        # Take the average of the 2 centers
        MinYCenter=(MinYCenter_1+MinYCenter_2)/2.0
        print("min y center values")
        print(MinYCenter_1,MinYCenter_2)
        
        self.slm.LCOS_Clean(self.channel)
        self.CamObj.SetContinousFrameCapMode(self.CamObj.Exposure.value)

        return int(MinXCenter),int(MinYCenter)
    
  
    def ChangePiFlipTake_digiholo(self,xVal):
        if(self.flipdir==0):
            xVal,x1Idx=AlignFunc.CovertCont2Desc(xVal,self.DescreteSpaceArrayX);
        else:
            xVal,x1Idx=AlignFunc.CovertCont2Desc(xVal,self.DescreteSpaceArrayY);
            
        Nx=self.slm.masksize[0]
        Ny=self.slm.masksize[1]
        # set up at the boundaries of the mask properties
        x_center_Input=int(self.slm.AllMaskProperties[self.channel][self.pol][self.imask].center[1])
        y_center_Input=int(self.slm.AllMaskProperties[self.channel][self.pol][self.imask].center[0])
  
        self.slm.AllMaskProperties[self.channel][self.pol][self.imask].zernike.zern_coefs[0]=0
        
        MASK=np.zeros((Nx,Ny),dtype=complex)
        #Left to right sweep
        if(self.flipdir==1):
            if( self.Phasedir==0):
                # self.PiFlip_cmplx[0:xVal+1,:]=np.exp(1j*np.pi)
                MASK[:,0:int((Nx/2))]=np.exp(1j*0)
                MASK[:,int((Nx/2)):Nx]=np.exp(1j*np.pi)
            else:
                MASK[:,0:int((Nx/2))]=np.exp(1j*np.pi)
                MASK[:,int((Nx/2)):Nx]=np.exp(1j*0)
                # self.PiFlip_cmplx[xVal:,:]=np.exp(1j*np.pi)
                
            if (self.ApplyZernike):
                MASK_PlussZernike=self.slm.ApplyZernikesToMask(self.channel,np.angle(MASK),imask=0,pol=self.pol,ipol=1,imode=0)
            else:
                MASK_PlussZernike=np.angle(MASK)
            MASKTODisplay=self.slm.Draw_Single_Mask( xVal, y_center_Input, MASK_PlussZernike)

        else:
            if( self.Phasedir==0):
                MASK[0:int((Ny/2)),:]=np.exp(1j*0)
                MASK[int((Ny/2)):Ny,:]=np.exp(1j*np.pi)
            else:
                MASK[0:int((Ny/2)),:]=np.exp(1j*np.pi)
                MASK[int((Ny/2)):Ny,:]=np.exp(1j*0)
                
            if (self.ApplyZernike):
                    MASK_PlussZernike=self.slm.ApplyZernikesToMask(self.channel,np.angle(MASK),imask=0,pol=self.pol,ipol=1,imode=0)
            else:
                MASK_PlussZernike=np.angle(MASK)
            MASKTODisplay=self.slm.Draw_Single_Mask( x_center_Input,xVal, MASK_PlussZernike)
            
        MASKTODisplay_256 = self.slm.phaseTolevel(MASKTODisplay)# Note that the -1*np.pi is so that the background is set to black it really doesn't matter though.
        # Display on SLM
        # slm.LCOS_Display(slm.LCOS_Screen_temp.astype(int), ch = 0)
        self.slm.LCOS_Display(MASKTODisplay_256, self.channel)
        
        #Going to go through and grab a few frames to get a average overlap with the ref Field to make it a but more acutrate 
        overlap_avgPWR=0
        for iframe in range(self.AvgFrameCount):
            # print('test')
            self.CamObj.GetFrame()
            Frame=self.CamObj.FrameBuffer.astype(np.float32)
            digholoFuncWrapper.ProcessBatchOfFrames(self.handleIdx,1,Frame)
            Field_Sig= digholoFuncWrapper.GetField(self.handleIdx)
            overlap=OpticOp.overlap(np.squeeze(Field_Sig),np.squeeze(np.conj(self.Field_Ref)))
            # print(overlap)

            overlap_avgPWR=overlap_avgPWR+(np.sum(np.abs(overlap)**2))

        RefSigPWR=np.sqrt(overlap_avgPWR/self.AvgFrameCount)
        RefSigPWR = np.sqrt(RefSigPWR)# not sure why I sqrt twice might be wrong will come back and check
        RefSigPWR_log = 10 * np.log10(RefSigPWR / self.RefPWR)
        
        
       
        return xVal,RefSigPWR_log
        



def CourseSweepAcrossSLM_digholo(slm:pyLCOS.LCOS,channel,CamObj:CamForm.GeneralCameraObject,flipCount,PixelsFromCenter):
     # need to set the camera to singleFrameCapturemode
    # CamObj.Exposure
    CamObj.SetSingleFrameCapMode()
    
    imask=0
    pol='V'
    PixelSize=CamObj.PixelSize
    #Centre wavelength (nanometres)
    lambda0 = 1550e-9
    #Polarisation components per frame
    polCount = 1 
    #Width/height of window to FFT on the camera. (pixels)
    # nx = 256
    # ny = 256
    CamDims=CamObj.FrameBuffer.shape
    frameWidth = CamDims[1]
    frameHeight = CamDims[0]
    nx = 320
    ny = 320
    # nx = 256
    # ny = 256
    #Amount of detail to print to console. 0: Console off. 1: Basic info. 2:Debug mode. 3: You've got serious issues
    verbosity = 0
    #Sets the resolution mode of the reconstructed field.
    #0 : Full resolution. Reconstructed field will have same pixel
    #size/dimensions as the FFT window.
    #1 : Low resolution. Reconstructed field will have dimension of the IFFT
    #window. 
    resolutionMode = 0
    consoleRedirectToFile=True
    maxMG=1
    AutoAlginFlagsCount=5
    AutoAlginFlags = np.zeros(AutoAlginFlagsCount,dtype=int)
    for i in range(AutoAlginFlagsCount):
        if (i==AutoAlginFlagsCount-1):
            AutoAlginFlags[i]=0
        else:
            AutoAlginFlags[i]=1
    fftWindowSizeX=256
    fftWindowSizeY=256
    handleIdx=digH_hpy.digHolo.digHoloCreate()

    #Clear the SLM
    slm.LCOS_Clean(channel)
    time.sleep(slm.GLobProps[channel].RefreshTime)
    #Get a frame for to make the reference field
    CamObj.GetFrame()
    FrameBufferInitial=CamObj.FrameBuffer.astype(np.float32)
    frameBufferPtr_initial = FrameBufferInitial.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    batchCount=1
    FFTRadius=0.4
    digH_hpy.digHolo.digHoloConfigSetFrameDimensions(handleIdx,frameWidth,frameHeight)
    digH_hpy.digHolo.digHoloConfigSetFramePixelSize(handleIdx,PixelSize)
    digH_hpy.digHolo.digHoloConfigSetBatchCount(handleIdx,batchCount) 
    digholoFuncWrapper.digholoSetProps(handleIdx,polCount,fftWindowSizeX,fftWindowSizeY,FFTRadius,lambda0,resolutionMode,maxMG,AutoAlginFlags)
    digholoFuncWrapper.digholo_AutoAlginBatch(handleIdx,batchCount,polCount,
                                                       fftWindowSizeX,fftWindowSizeY,FFTRadius,lambda0,maxMG,resolutionMode,verbosity,
                                                       AutoAlginFlags,frameBufferPtr_initial)
    digholoFuncWrapper.ProcessBatchOfFrames(handleIdx,batchCount,FrameBufferInitial)
    #This is the reference Field that the other fields will be overlaped with
    Field_Ref=(digholoFuncWrapper.GetField(handleIdx))
    RefPWR=np.sum(np.abs(OpticOp.overlap(Field_Ref,np.conj(Field_Ref)))**2)
    plt.imshow(cmplxplt.ComplexArrayToRgb(np.squeeze(Field_Ref)));
    
    imask=0
    pol='V'
    
  
    # set up at the boundaries of the mask properties
    x_center=slm.AllMaskProperties[channel][pol][imask].center[1]
    y_center=slm.AllMaskProperties[channel][pol][imask].center[0]
    print(x_center,y_center)

    flipMinX=x_center-PixelsFromCenter
    if flipMinX<0:
        flipMinX=0
    flipMaxX=x_center+PixelsFromCenter
    if flipMaxX>slm.slmWidth:
        flipMaxX=slm.slmWidth-1
    flipMinY=y_center-PixelsFromCenter
    if flipMinY<0:
        flipMinY=0
    flipMaxY=y_center+PixelsFromCenter
    if flipMaxY>slm.slmHeigth:
        flipMaxY=slm.slmHeigth-1
        
    # flipMaxY=slm.slmHeigth//2+flipCount//2
    # flipMax=slm.slmWidth//2+flipCount//2
    RefSigPWR_X=np.empty(0)
    RefSigPWR_Y=np.empty(0)
    
    powerReadingX=np.empty(0)
    # CountX=np.empty(0)
    PixelFlipX=np.empty(0)
    powerReadingY=np.empty(0)
    # CountY=np.empty(0)
    PixelFlipY=np.empty(0)
    AvgFrameCount=1

    #Left to right sweep
    i=0
    for iflip in range(flipMinX,flipMaxX,flipCount):
        PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)*np.exp(0.0*1j*np.pi)
        # PiFlip_cmplx =np.zeros((slm.slmHeigth,slm.slmWidth),dtype=np.float32)
        # PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=np.float32)*(-1*np.pi)

        # PiFlip_cmplx[0:flipMin+iflip,:]=np.exp(1j*np.pi)
        PiFlip_cmplx[:,0:iflip]=PiFlip_cmplx[:,0:iflip]*np.exp(1j*np.pi)
        # PiFlip_cmplx[:,0:iflip]=(np.pi)


        # np.angle( np.random.rand(1200,1920) + np.random.rand(1200,1920) * 1j)
        ArryForSLM=slm.phaseTolevel(np.angle(PiFlip_cmplx))
        # slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
        slm.LCOS_Display(ArryForSLM, channel)
        
        
        # time.sleep(slm.GLobProps[channel].RefreshTime)
        # time.sleep(0.5)

        PixelFlipX=np.append(PixelFlipX,iflip)

        #Going to go through and grab a few frames to get a average overlap with the ref Field to make it a but more acutrate 
        overlap_avgPWR=0
        for iframe in range(AvgFrameCount):
            # print('test')
            # CamObj.GetFrame()
            # Frame=CamObj.FrameBuffer.astype(np.float32)
            Frame=CamObj.GetFrame()
            digholoFuncWrapper.ProcessBatchOfFrames(handleIdx,1,Frame.astype(np.float32))
            Field_Sig= digholoFuncWrapper.GetField(handleIdx)
            overlap=OpticOp.overlap(np.squeeze(Field_Sig),np.squeeze(np.conj(Field_Ref)))
            # print(overlap)

            overlap_avgPWR=overlap_avgPWR+(np.sum(np.abs(overlap)**2))
        if(i==flipCount//2):
                fieldmid=copy.deepcopy(Field_Sig)
                cammid=copy.deepcopy(Frame)
        i=i+1
    
        RefSigPWR_X=np.append(RefSigPWR_X,np.sqrt(overlap_avgPWR/AvgFrameCount))# work out the average overlap and put it in the RefSigPWR_X array


    RefSigPWR_X = np.sqrt(RefSigPWR_X)# not sure why I sqrt twice might be wrong will come back and check
    RefSigPWR_log_X = 10 * np.log10(RefSigPWR_X / RefPWR)

        
        
    # top to bottom sweep    
    i=0
    for iflip in range(flipMinY,flipMaxY,flipCount):

        PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)*np.exp(0.0*1j*np.pi)
        # PiFlip_cmplx =np.zeros((slm.slmHeigth,slm.slmWidth),dtype=np.float32)

        PiFlip_cmplx[0:iflip,:]= PiFlip_cmplx[0:iflip,:]*np.exp(1j*np.pi)
        # PiFlip_cmplx[0:flipMin+iflip,:]=(np.pi)

        # PiFlip_cmplx[:,0:flipMin+iflip]=np.exp(1j*np.pi)

        # np.angle( np.random.rand(1200,1920) + np.random.rand(1200,1920) * 1j)
        ArryForSLM=slm.phaseTolevel(np.angle(PiFlip_cmplx))
        # slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
        slm.LCOS_Display(ArryForSLM,channel)
        
        # time.sleep(slm.GLobProps[channel].RefreshTime)
        PixelFlipY=np.append(PixelFlipY,iflip)
        
          #Going to go through and grab a few frames to get a average overlap with the ref Field to make it a but more acutrate 
        overlap_avgPWR=0
        for iframe in range(AvgFrameCount):
            # CamObj.GetFrame()
            # Frame=CamObj.FrameBuffer.astype(np.float32)
            # digholoFuncWrapper.ProcessBatchOfFrames(handleIdx,1,Frame)
            Frame=CamObj.GetFrame()
            digholoFuncWrapper.ProcessBatchOfFrames(handleIdx,1,Frame.astype(np.float32))
            Field_Sig= digholoFuncWrapper.GetField(handleIdx)            
            overlap=OpticOp.overlap(np.squeeze(Field_Sig),np.squeeze(np.conj(Field_Ref)))
            overlap_avgPWR=overlap_avgPWR+(np.sum(np.abs(overlap)**2))
        i=i+1
    
        RefSigPWR_Y=np.append(RefSigPWR_Y,np.sqrt(overlap_avgPWR/AvgFrameCount))# work out the average overlap and put it in the RefSigPWR_X array


    RefSigPWR_Y = np.sqrt(RefSigPWR_Y)# not sure why I sqrt twice might be wrong will come back and check
    RefSigPWR_log_Y = 10 * np.log10(RefSigPWR_Y / RefPWR)
    
    slm.LCOS_Clean(channel)

    ErrorCode=digH_hpy.digHolo.digHoloDestroy(handleIdx) 
    print(ErrorCode)
    CamObj.SetContinousFrameCapMode(CamObj.Exposure.value)
    return RefSigPWR_X,RefSigPWR_log_X,PixelFlipX,RefSigPWR_Y,RefSigPWR_log_Y,PixelFlipY,fieldmid,cammid

