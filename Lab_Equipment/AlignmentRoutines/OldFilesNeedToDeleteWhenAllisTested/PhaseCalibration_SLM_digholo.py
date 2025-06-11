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

# from script_functions import start_worker
# import CameraWindowForm as CamForm
#SLM Libs
import Lab_Equipment.SLM.pyLCOS as pyLCOS
import Lab_Equipment.ZernikeModule.ZernikeModule as zernMod

#Camera Libs
import Lab_Equipment.Camera.CameraObject as CamForm

# digiHolo Libs
import Experiments.Lab_Equipment.digHolo.digHolo_pylibs.digiholoHeader as digH_hpy  # as in header file for python... pretty clever I know (Daniel 2 seconds after writing this commment. Head slap you are a idiot )
import Lab_Equipment.digHolo.digHolo_pylibs.digiholoWindowForm as digForm
import Lab_Equipment.digHolo.digHolo_pylibs.digholoCombinedFunction as digholoFuncWrapper

# Daniel's Python Libs
import  Lab_Equipment.MyPythonLibs.OpticalOperators as OpticOp
import  Lab_Equipment.MyPythonLibs.ComplexPlotFunction as cmplxplt
import  Lab_Equipment.MyPythonLibs.GaussianBeamBasis as GaussBeams
import  Lab_Equipment.MyPythonLibs.GeneralFunctions as GenFuncs
import  Lab_Equipment.MyPythonLibs.AnalysisFuncitons as ProCouplMat
import  Lab_Equipment.MyPythonLibs.CoupMatrixAndMetricAnalysisFuncitons as MetricCals
import  Lab_Equipment.MyPythonLibs.SaveMaskToBinFile as SaveMaskBin
import  Lab_Equipment.MyPythonLibs.ModelabProcessBatchFile as Modelab


def ProcessFramesFromPhaseCal(FrameBuffer,MaskCount,FFTRadiusIn,wavelength=1550e-9,Nx=256,Ny=256,CampixelSize=30e-6):
    frameWidth = ((ctypes.c_int))()
    frameHeight = ((ctypes.c_int))()
    frameCount = ((ctypes.c_int))()
    batchCount = ((ctypes.c_int))()
    polCount = ((ctypes.c_int))()
    fftWindowSizeY= ((ctypes.c_int))()
    fftWindowSizeX = ((ctypes.c_int))()
    maxMG= ((ctypes.c_int))()
    resolutionMode=((ctypes.c_int))()

    pixelSize =((ctypes.c_float))()
    lambda0 =((ctypes.c_float))()



    FrameDims=FrameBuffer.shape
    phaseLevels=FrameDims[0]-1
    mplcIdx=0
    # Setup alginment stuff for digholo to get the fields
    pixelSize = CampixelSize
    #Centre wavelength (nanometres)
    lambda0 = wavelength
    #Polarisation components per frame
    polCount = 1
    #Width/height of window to FFT on the camera. (pixels)
    # nx = 256
    # ny = 256
    CamDims=FrameBuffer.shape
    frameWidth = CamDims[2]
    frameHeight = CamDims[1]
    # nx = 512
    # ny = 512
    # nx = 256
    # ny = 256
    #Amount of detail to print to console. 0: Console off. 1: Basic info. 2:Debug mode. 3: You've got serious issues
    verbosity = 2
    #Sets the resolution mode of the reconstructed field.
    #0 : Full resolution. Reconstructed field will have same pixel
    #size/dimensions as the FFT window.
    #1 : Low resolution. Reconstructed field will have dimension of the IFFT
    #window. 
    resolutionMode = 0
    AutoAlginFlagsCount=5
    AutoAlginFlags = np.zeros(AutoAlginFlagsCount,dtype=int)
    for i in range(AutoAlginFlagsCount):
        if (i==AutoAlginFlagsCount-1):
            AutoAlginFlags[i]=0
        else:
            AutoAlginFlags[i]=1

    consoleRedirectToFile=True
    maxMG=1
    handleIdx=digH_hpy.digHolo.digHoloCreate()
    # FrameBufferInitial = camera.Get_Singleframe()
    FrameBufferInitial= copy.deepcopy(FrameBuffer[-1,:,:])
    frameBufferPtr_initial = FrameBufferInitial.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    fftWindowSizeX=Nx
    fftWindowSizeY=Ny
    batchCount=1
    FFTRadius=FFTRadiusIn
    digH_hpy.digHolo.digHoloConfigSetFrameDimensions(handleIdx,frameWidth,frameHeight)
    digH_hpy.digHolo.digHoloConfigSetFramePixelSize(handleIdx,pixelSize)
    digH_hpy.digHolo.digHoloConfigSetBatchCount(handleIdx,batchCount) 
    digholoFuncWrapper.digholoSetProps(handleIdx,polCount,fftWindowSizeX,fftWindowSizeY,FFTRadius,lambda0,resolutionMode,maxMG,AutoAlginFlags)

    # digholoFuncWrapper.digholo_Initialise(handleIdx,batchCount,frameBufferPtr_initial,FFTRadius,pixelSize,lambda0,polCount,nx,ny,verbosity,resolutionMode,consoleRedirectToFile,frameWidth,frameHeight,maxMG)
    digholoFuncWrapper.digholo_AutoAlginBatch(handleIdx,batchCount,polCount,
                                                       fftWindowSizeX,fftWindowSizeY,FFTRadius,lambda0,maxMG,resolutionMode,verbosity,
                                                       AutoAlginFlags,frameBufferPtr_initial)
    digholoFuncWrapper.ProcessBatchOfFrames(handleIdx,batchCount,FrameBufferInitial)
    Fullimage ,ViewPortRGB_cam,WindowString=digholoFuncWrapper.GetViewport_arr(handleIdx,FrameBufferInitial)
    plt.imshow(Fullimage)

    batchCount= phaseLevels
    digholoFuncWrapper.ProcessBatchOfFrames(handleIdx,batchCount,FrameBuffer[0:-1,:,:])
    Fields= digholoFuncWrapper.GetField(handleIdx)
    NewFileForBatch="phaseCal_" + str(1550) + "_MPLC_" +str(mplcIdx) + "_bounce_" + str(MaskCount)
    digholoFuncWrapper.SaveBatchFile(NewFileForBatch,handleIdx,FrameBuffer[0:-1,:,:],True)

    ErrorCode=digH_hpy.digHolo.digHoloDestroy(handleIdx) 
    print(ErrorCode)
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
    mplcIdx=0
    MaskCount=slm.MaskCount
    Nx=slm.masksize[0]
    Ny=slm.masksize[1]
    
    Centerloc=550
    slm.AllMaskProperties[channel][pol][imask].zernike.zern_coefs[0]=0 # if the piston is not set to zero set it to zero
    y_center = slm.AllMaskProperties[channel][pol][imask].center[0]
    x_center = slm.AllMaskProperties[channel][pol][imask].center[1]   
    MaskSize = slm.masksize
    
    
    FrameBuffer = np.zeros((phaseLevels+1, CamObj.FrameHeight, CamObj.FrameWidth), dtype=np.float32)
    FlipCenter=Centerloc
    slm_array = np.zeros((slm.slmHeigth,slm.slmWidth), dtype=int)
    # phaseLevel_257=np.linspace(-np.pi,np.pi,257)
    phaseLevel_257=np.linspace(0,2*np.pi,257)

    phaseLevel=phaseLevel_257[0:-1]
    print(np.shape(phaseLevel))
    MASK=np.zeros((Nx,Ny),dtype=complex)
    for level in range(phaseLevels):
        print(level, end=' ')
        # Create phase wrap 
        # MASK[:,0:int((Nx/2))]=128
        # MASK[:,int((Nx/2)):Nx]=level
        # MASK[0:int((Ny/2)),:]=128
        # MASK[int((Ny/2)):Ny,:]=level
        if(Direction=="y"):
            MASK[0:int((Ny/2)),:]=np.exp(1j*phaseLevel[0])
            MASK[int((Ny/2)):Ny,:]=np.exp(1j*phaseLevel[level])
        elif(Direction=="x"):
            MASK[:,0:int((Nx/2))]=np.exp(1j*phaseLevel[0])
            MASK[:,int((Nx/2)):Nx]=np.exp(1j*phaseLevel[level])
            
        MASK_PlussZernike=slm.ApplyZernikesToMask(channel,np.angle(MASK),imask=0,pol=pol,ipol=1,imode=0)
        MASKTODisplay=slm.Draw_Single_Mask( x_center, y_center, MASK_PlussZernike)
        MASKTODisplay_256 = slm.phaseTolevel(MASKTODisplay)# Note that the -1*np.pi is so that the background is set to black it really doesn't matter though.
        # Display on SLM
        # slm.LCOS_Display(slm.LCOS_Screen_temp.astype(int), ch = 0)
        # slm.LCOS_Display(MASKTODisplay_256, channelIdx=slm.GLobProps[channel].rgbChannelIdx)
        slm.LCOS_Display(MASKTODisplay_256,channel)
        
        time.sleep(slm.GLobProps[channel].RefreshTime)  
        CamObj.GetFrame()
        CamFrame=CamObj.FrameBuffer.astype(np.float32)
    
        FrameBuffer[level,:,:] = CamFrame

    MASK[:,:]=np.exp(1j*phaseLevel[0])
    # MASK[int((Ny/2)):Ny,:]=np.exp(1j*phaseLevel[0])
    MASK_PlussZernike=slm.ApplyZernikesToMask(channel,np.angle(MASK),imask=0,pol=pol,ipol=1,imode=0)
    MASKTODisplay=slm.Draw_Single_Mask( x_center, y_center, MASK_PlussZernike)
    MASKTODisplay_256 = slm.phaseTolevel(MASKTODisplay)# Note that the -1*np.pi is so that the background is set to black it really doesn't matter though.
    # Display on SLM
    slm.LCOS_Display(MASKTODisplay_256, channel)
    
    time.sleep(slm.GLobProps[channel].RefreshTime)  
    CamObj.GetFrame()
    CamFrame=CamObj.FrameBuffer.astype(np.float32)
    
    # slm.LCOS_Clean()    
    # time.sleep(slm.GLobProps.RefreshTime*4) 
     
    # CamObj.GetFrame()
    # CamFrame=CamObj.FrameBuffer.astype(np.float32)
    FrameBuffer[-1,:,:] = CamFrame
    #Turn continous mode back on for the camera
    CamObj.SetContinousFrameCapMode(CamObj.Exposure)


    return FrameBuffer
