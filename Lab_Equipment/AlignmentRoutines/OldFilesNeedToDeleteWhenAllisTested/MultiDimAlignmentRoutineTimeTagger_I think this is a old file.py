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
# import PyCapture2 # module for the camera
if (config.PYCAPTURE_IMPORT):
    import PyCapture2
else:
    from vmbpy import *
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

# Daniel's Python Libs
import  Lab_Equipment.MyPythonLibs.OpticalOperators as OpticOp
import  Lab_Equipment.MyPythonLibs.ComplexPlotFunction as cmplxplt
import  Lab_Equipment.MyPythonLibs.GaussianBeamBasis as GaussBeams
import  Lab_Equipment.MyPythonLibs.GeneralFunctions as GenFuncs
import  Lab_Equipment.MyPythonLibs.AnalysisFuncitons as ProCouplMat
import  Lab_Equipment.MyPythonLibs.CoupMatrixAndMetricAnalysisFuncitons as MetricCals
import  Lab_Equipment.MyPythonLibs.SaveMaskToBinFile as SaveMaskBin
import  Lab_Equipment.MyPythonLibs.ModelabProcessBatchFile as Modelab
import Lab_Equipment.TimeTagger.TimeTaggerInterface as TimeTaggerMod

# Ok so i dont know if this will work but I am trying to make a generic golden search function that i can
# pass a function too that is described in a class.
# I can confirm that it worked. Essentiall you make class that has the function in it and as long as the function only has on input and one output it will work 

class CenterAlginmentSpace():
    def __init__(self,slmObject:pyLCOS.LCOS,slmChannel ,TtaggerObj:TimeTaggerMod):
        super().__init__()
        self.slm=slmObject
        self.timeTagger=TtaggerObj
        self.PiFlip_cmplx =np.ones((self.slm.LCOSsize[0],self.slm.LCOSsize[1]),dtype=complex)
        # PiFlip_cmplx[:,:]=np.exp(1j*np.pi)
        self.flipdir=0
        self.Phasedir=0
        self.channel=slmChannel
        self.imask=0
        self.pol='V'
        self.ApplyZernike=True
        
        #set the time tagger to single capture
        self.timeTagger.setSingleCaptureMode()
        self.timeTagger.setCountingTime(0.1*1e12)
        self.avgCount=10
    def PerformCenterAlignment(self,TotalSpaceArrX,TotalSpaceArrY):
        
        self.flipdir=0 # flipdir X
        #Need to do 2 flips in the same direction for a better center one has the flip reversed
        self.Phasedir=0
        minValX_1,minIdxX_1=GoldenSelectionSearch(0,-1,TotalSpaceArrX,self.ChangePiFlipTakeCoincidence)
        MinXCenter_1=TotalSpaceArrX[minIdxX_1]
        # reversed flip in same direction
        self.Phasedir=1
        minValX_2,minIdxX_2=GoldenSelectionSearch(0,-1,TotalSpaceArrX,self.ChangePiFlipTakeCoincidence)
        MinXCenter_2=TotalSpaceArrX[minIdxX_2]
        # Take the average of the 2 centers
        MinXCenter=(MinXCenter_1+MinXCenter_2)/2.0

        print("min x center values")
        print(MinXCenter_1,MinXCenter_2)
        
        #Now we will go in the the other direction, again you need to do 2 flips for a better results 
        #Need to do 2 flips in the same direction for a better center one has the flip reversed
        self.flipdir=1
        self.Phasedir=0
        minValY_1,minIdxY_1=GoldenSelectionSearch(0,-1,TotalSpaceArrY,self.ChangePiFlipTakeCoincidence)
        MinYCenter_1=TotalSpaceArrY[minIdxY_1]
        # reversed flip in same direction
        self.Phasedir=1
        minValY_2,minIdxY_2=GoldenSelectionSearch(0,-1,TotalSpaceArrY,self.ChangePiFlipTakeCoincidence)
        MinYCenter_2=TotalSpaceArrY[minIdxY_2]
        # Take the average of the 2 centers
        MinYCenter=(MinYCenter_1+MinYCenter_2)/2.0
        print("min y center values")
        print(MinYCenter_1,MinYCenter_2)
        self.slm.LCOS_Clean(self.channel)
        self.timeTagger.setContinuousCaptureMode()

        return MinXCenter,MinYCenter
        
    
    def ChangePiFlipTakeCoincidence(self,xVal):
      
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
        
        #grab avgCount coincidences measurments and take the average to work out the average coincidences
        CoinAvg=0
        countAvg=0
        for iavg in range(self.avgCount):
            CoinData=self.timeTagger.getCoincidences()# get timetagger coincidenece
            CoinAvg=CoinAvg+CoinData[2]
            countAvg=countAvg+CoinData[0]
            
            

        # self.PiFlip_cmplx =np.ones((self.slm.LCOSsize[0],self.slm.LCOSsize[1]),dtype=complex)
        # if(self.flipdir==0):
        #     if( self.Phasedir==0):
        #         self.PiFlip_cmplx[0:xVal+1,:]=np.exp(1j*np.pi)
        #     else:
        #         self.PiFlip_cmplx[xVal:,:]=np.exp(1j*np.pi)

        # else:
        #     if( self.Phasedir==0):
        #         self.PiFlip_cmplx[:,0:xVal+1]=np.exp(1j*np.pi)
        #     else:
        #         self.PiFlip_cmplx[:,xVal:]=np.exp(1j*np.pi)

 
        # ArryForSLM=self.slm.phaseTolevel(np.angle(self.PiFlip_cmplx))
        # self.slm.LCOS_Display(ArryForSLM, channelIdx = self.slm.GLobProps[self.channel].rgbChannelIdx)

        # #grab avgCount coincidences measurments and take the average to work out the average coincidences
        # CoinAvg=0
        # for iavg in range( self.avgCount):
        #     CoinData=self.timeTagger.getCoincidences()# get timetagger coincidenece
        #     CoinAvg=CoinAvg+CoinData[2]
       
        return CoinAvg/self.avgCount
        


def CovertCont2Desc(contValue,Distarr):
    N=np.size(Distarr)
    FoundIdx=False
    i=0
    while(FoundIdx != True):
        if(i<N-1): 
            if(contValue >= Distarr[i] and contValue < Distarr[i+1]):
                if (contValue >= (Distarr[i] + Distarr[i+1])/2):
                    DistValue = Distarr[i+1]
                    DistIdx = i+1
                else:
                    DistValue = Distarr[i]
                    DistIdx = i
                FoundIdx=True
        else:#Need to consider the last value
            if (contValue >= (Distarr[i-1] + Distarr[i])/2):
                    DistValue = Distarr[i]
                    DistIdx = i
            else:
                DistValue = Distarr[i-1]
                DistIdx = i-1
            FoundIdx=True
        i=i+1
        if(i>N):#We will cap any values that are outside the range of to the discrecte values
            print("Bad value", contValue)
            DistValue = Distarr[N-1]
            DistIdx = N-1
    return DistValue, DistIdx   

# Program 13.1 Golden Section Search for minimum of f(x)
# Start with unimodal f(x) and minimum in [a,b]
# Input: function f, interval [a,b], number of steps k
# Output: approximate minimum y
def GoldenSelectionSearch(aIdx,bIdx,TotalSpaceArr,FuncToMinamise):
    xArr= TotalSpaceArr#= FieldSuperPixel(FieldToProb, PixelIdxRange,PhaseValue)
    iNumIterations=0
    goldenRation=(np.sqrt(5)-1)/2;
    
    # Lets just do a course check on the problem space and see if we can get close to the root to reduce the number of calls to the function
    #aIdx,bIdx=CourseSearchOptimisation(aIdx,bIdx,ProblemProps,MinFunc)
    
    a=xArr[aIdx]
    b=xArr[bIdx]

    x1 = a+(1-goldenRation)*(b-a);
    x1,x1Idx=CovertCont2Desc(x1,xArr);
    x2 = a+goldenRation*(b-a);
    x2,x2Idx=CovertCont2Desc(x2,xArr);
    
    f1=FuncToMinamise(xArr[x1Idx]);
    f2=FuncToMinamise(xArr[x2Idx]);
    print(f1,f2)
    dspace = np.abs(x1Idx-x2Idx);
    dspace_Tol=1;
    f_aAndb=np.zeros(2)
    f_aAndbIdx=np.zeros(2,dtype=int)
    #print("New Super pixel")
    #for i in range(k):#1:k
    while(dspace > dspace_Tol):
        if (f1 < f2): # if f(x1) < f(x2), replace b with x2
            b=x2; 
            x2=x1; 
            x1=a+(1-goldenRation)*(b-a);
            x1,x1Idx=CovertCont2Desc(x1,xArr);
            f2=f1; 
            f1=FuncToMinamise(xArr[x1Idx]);  # single function evaluation
        else: #otherwise, replace a with x1
            a=x1; 
            x1=x2 
            x2 = a+goldenRation*(b-a);
            x2,x2Idx=CovertCont2Desc(x2,xArr);
            f1=f2; 
            f2=FuncToMinamise(xArr[x2Idx]);  # single function evaluation
        
        dspace = abs(x2Idx-x1Idx)
        print(f1,x1,dspace)
        iNumIterations=iNumIterations+1 
          
    #Work out which one is the best value to take. This is kind of doing a y=(a+b)/2 but since we are dealing with integer we are not doing a half thing we are 
    #just working out which one is a the lowest
    a,aIdx=CovertCont2Desc(a,xArr);
    b,bIdx=CovertCont2Desc(b,xArr);
    f_aAndb[0]=FuncToMinamise(xArr[aIdx])
    f_aAndb[1]=FuncToMinamise(xArr[bIdx])
    f_aAndbIdx[0]=int(aIdx)
    f_aAndbIdx[1]=int(bIdx)
    minIdx=np.argmin(f_aAndb)
    minValue=np.min(f_aAndb)
    #print(iNumIterations)
    
    return -1*minValue,f_aAndbIdx[minIdx]


class MultiDimAlginmentSpace():
    def __init__(self,slmObject:pyLCOS.LCOS,slmChannel ,TtaggerObj:TimeTaggerMod):
        super().__init__()
        self.slm=slmObject
        self.timeTagger=TtaggerObj
        self.PiFlip_cmplx =np.ones((self.slm.LCOSsize[0],self.slm.LCOSsize[1]),dtype=complex)
        # PiFlip_cmplx[:,:]=np.exp(1j*np.pi)
        self.flipdir=0
        self.Phasedir=0
        self.channel=slmChannel
        self.imask=0
        self.pol='V'
        self.ApplyZernike=True
        
        #set the time tagger to single capture
        self.timeTagger.setSingleCaptureMode()
        self.timeTagger.setCountingTime(0.1*1e12)
        self.avgCount=10
    def PerformCenterAlignment(self,TotalSpaceArrX,TotalSpaceArrY):
        
        self.flipdir=0 # flipdir X
        #Need to do 2 flips in the same direction for a better center one has the flip reversed
        self.Phasedir=0
        minValX_1,minIdxX_1=GoldenSelectionSearch(0,-1,TotalSpaceArrX,self.ChangePiFlipTakeCoincidence)
        MinXCenter_1=TotalSpaceArrX[minIdxX_1]
        # reversed flip in same direction
        self.Phasedir=1
        minValX_2,minIdxX_2=GoldenSelectionSearch(0,-1,TotalSpaceArrX,self.ChangePiFlipTakeCoincidence)
        MinXCenter_2=TotalSpaceArrX[minIdxX_2]
        # Take the average of the 2 centers
        MinXCenter=(MinXCenter_1+MinXCenter_2)/2.0

        print("min x center values")
        print(MinXCenter_1,MinXCenter_2)
        
        #Now we will go in the the other direction, again you need to do 2 flips for a better results 
        #Need to do 2 flips in the same direction for a better center one has the flip reversed
        self.flipdir=1
        self.Phasedir=0
        minValY_1,minIdxY_1=GoldenSelectionSearch(0,-1,TotalSpaceArrY,self.ChangePiFlipTakeCoincidence)
        MinYCenter_1=TotalSpaceArrY[minIdxY_1]
        # reversed flip in same direction
        self.Phasedir=1
        minValY_2,minIdxY_2=GoldenSelectionSearch(0,-1,TotalSpaceArrY,self.ChangePiFlipTakeCoincidence)
        MinYCenter_2=TotalSpaceArrY[minIdxY_2]
        # Take the average of the 2 centers
        MinYCenter=(MinYCenter_1+MinYCenter_2)/2.0
        print("min y center values")
        print(MinYCenter_1,MinYCenter_2)
        self.slm.LCOS_Clean(self.channel)
        self.timeTagger.setContinuousCaptureMode()

        return MinXCenter,MinYCenter
        
    
    def ChangePiFlipTakeCoincidence(self,xVal):
      
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
        
        #grab avgCount coincidences measurments and take the average to work out the average coincidences
        CoinAvg=0
        countAvg=0
        for iavg in range(self.avgCount):
            CoinData=self.timeTagger.getCoincidences()# get timetagger coincidenece
            CoinAvg=CoinAvg+CoinData[2]
            countAvg=countAvg+CoinData[0]
            
            

        # self.PiFlip_cmplx =np.ones((self.slm.LCOSsize[0],self.slm.LCOSsize[1]),dtype=complex)
        # if(self.flipdir==0):
        #     if( self.Phasedir==0):
        #         self.PiFlip_cmplx[0:xVal+1,:]=np.exp(1j*np.pi)
        #     else:
        #         self.PiFlip_cmplx[xVal:,:]=np.exp(1j*np.pi)

        # else:
        #     if( self.Phasedir==0):
        #         self.PiFlip_cmplx[:,0:xVal+1]=np.exp(1j*np.pi)
        #     else:
        #         self.PiFlip_cmplx[:,xVal:]=np.exp(1j*np.pi)

 
        # ArryForSLM=self.slm.phaseTolevel(np.angle(self.PiFlip_cmplx))
        # self.slm.LCOS_Display(ArryForSLM, channelIdx = self.slm.GLobProps[self.channel].rgbChannelIdx)

        # #grab avgCount coincidences measurments and take the average to work out the average coincidences
        # CoinAvg=0
        # for iavg in range( self.avgCount):
        #     CoinData=self.timeTagger.getCoincidences()# get timetagger coincidenece
        #     CoinAvg=CoinAvg+CoinData[2]
       
        return CoinAvg/self.avgCount




def CourseSweepWithMasksSLMCoincidence(slm:pyLCOS.LCOS,channel,timeTagger:TimeTaggerMod,flipCount,PixelsFromCenter,ChangeCentersToNewCenters,ApplyZernikes):
    imask=0
    pol="V"
    # OrginalMask=slm.AllMaskProperties[channel][pol][imask].Mask[:,:,:]
    timeTagger.setSingleCaptureMode()
    timeTagger.setCountingTime(0.1*1e12)
    Nx=slm.masksize[0]
    Ny=slm.masksize[1]
    avgCount=10
    CoinAvg=0
    # set up at the boundaries of the mask properties
    x_center=int(slm.AllMaskProperties[channel][pol][imask].center[1])
    y_center=int(slm.AllMaskProperties[channel][pol][imask].center[0])
    x_center_Input=x_center
    y_center_Input=y_center
    slm.AllMaskProperties[channel][pol][imask].zernike.zern_coefs[0]=0 # if the piston is not set to zero set it to zero

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

    powerReadingX=np.empty(0)
    CountX=np.empty(0)
    PixelFlipX=np.empty(0)
    powerReadingY=np.empty(0)
    CountY=np.empty(0)
    PixelFlipY=np.empty(0)
    MASK=np.zeros((Nx,Ny),dtype=complex)
    #Left to right sweep
    for iflip in range(flipMinX,flipMaxX,flipCount):
        MASK[:,0:int((Nx/2))]=np.exp(1j*0)
        MASK[:,int((Nx/2)):Nx]=np.exp(1j*np.pi)
        if (ApplyZernikes):
            MASK_PlussZernike=slm.ApplyZernikesToMask(channel,np.angle(MASK),imask=0,pol=pol,ipol=1,imode=0)
        else:
            MASK_PlussZernike=np.angle(MASK)
        MASKTODisplay=slm.Draw_Single_Mask( iflip, y_center_Input, MASK_PlussZernike)
        MASKTODisplay_256 = slm.phaseTolevel(MASKTODisplay)# Note that the -1*np.pi is so that the background is set to black it really doesn't matter though.
        # Display on SLM
        # slm.LCOS_Display(slm.LCOS_Screen_temp.astype(int), ch = 0)
        slm.LCOS_Display(MASKTODisplay_256, channel) 
        # Work out the coincidences
        CoinAvg=0
        countAvg=0
        for iavg in range(avgCount):
            CoinData=timeTagger.getCoincidences()# get timetagger coincidenece
            CoinAvg=CoinAvg+CoinData[2]
            countAvg=countAvg+CoinData[0]

        print(CoinAvg/avgCount)
        CountX=np.append(CountX,countAvg/avgCount)
        powerReadingX=np.append(powerReadingX,CoinAvg/avgCount)
        PixelFlipX=np.append(PixelFlipX,iflip)
    
    
    for iflip in range(flipMinY,flipMaxY,flipCount):
        # slm.AllMaskProperties[channel][pol][imask].center[1]=x_center_Input
        # slm.AllMaskProperties[channel][pol][imask].center[0]=iflip
        MASK[0:int((Ny/2)),:]=np.exp(1j*0)
        MASK[int((Ny/2)):Ny,:]=np.exp(1j*np.pi)
        if (ApplyZernikes):
            MASK_PlussZernike=slm.ApplyZernikesToMask(channel,np.angle(MASK),imask=0,pol=pol,ipol=1,imode=0)
        else:
            MASK_PlussZernike=np.angle(MASK)
        MASKTODisplay=slm.Draw_Single_Mask( x_center_Input, iflip, MASK_PlussZernike)
        MASKTODisplay_256 = slm.phaseTolevel(MASKTODisplay)# Note that the -1*np.pi is so that the background is set to black it really doesn't matter though.
        # Display on SLM
        # slm.LCOS_Display(slm.LCOS_Screen_temp.astype(int), ch = 0)
        # slm.LCOS_Display(MASKTODisplay_256, channelIdx=slm.GLobProps[channel].rgbChannelIdx)
        slm.LCOS_Display(MASKTODisplay_256, channel)
         # Work out the coincidences
        countAvg=0
        CoinAvg=0
        for iavg in range( avgCount):
            CoinData=timeTagger.getCoincidences()# get timetagger coincidenece
            CoinAvg=CoinAvg+CoinData[2]
            countAvg=countAvg+CoinData[0]

        print(CoinAvg/avgCount)
        CountY=np.append(CountY,countAvg/avgCount)
        powerReadingY=np.append(powerReadingY,CoinAvg/avgCount)
        PixelFlipY=np.append(PixelFlipY,iflip)

    
    minIdxX=np.argmin(powerReadingX)
    minIdxY=np.argmin(powerReadingY)


    print("Minimum coincidence value",powerReadingX[minIdxX] ,"X Center =",PixelFlipX[minIdxX])
    print("Original X Center= ", x_center_Input)
    print("Minimum coincidence value",powerReadingY[minIdxY] ,"Y Center =",PixelFlipY[minIdxY])
    print("Original Y Center= ", y_center_Input)

    if(ChangeCentersToNewCenters):
        slm.AllMaskProperties[channel][pol][imask].center[0]=PixelFlipY[minIdxX]
        slm.AllMaskProperties[channel][pol][imask].center[1]=PixelFlipX[minIdxY]
    slm.LCOS_Clean(channel)
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(PixelFlipX,powerReadingX)
    plt.subplot(1,2,2)
    plt.plot(PixelFlipY,powerReadingY)
    timeTagger.setContinuousCaptureMode()
    return powerReadingX,CountX,PixelFlipX,powerReadingY,CountY,PixelFlipY

def CourseSweepAcrossSLMCoincidence(slm:pyLCOS.LCOS,channel,timeTagger:TimeTaggerMod,flipCount,PixelsFromCenter):
    imask=0
    pol='V'
    
    # slm.LCOS_Clean()
    timeTagger.setSingleCaptureMode()
    timeTagger.setCountingTime(0.3*1e12)
    avgCount=10
    CoinAvg=0
    for iavg in range( avgCount):
            CoinData=timeTagger.getCoincidences()# get timetagger coincidenece
            CoinAvg=CoinAvg+CoinData[0]
    print(CoinAvg/avgCount)
  
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
    
    powerReadingX=np.empty(0)
    CountX=np.empty(0)
    PixelFlipX=np.empty(0)
    powerReadingY=np.empty(0)
    CountY=np.empty(0)
    PixelFlipY=np.empty(0)

    #Left to right sweep
    for iflip in range(flipMinX,flipMaxX,flipCount):
        PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)
        # PiFlip_cmplx =np.zeros((slm.slmHeigth,slm.slmWidth),dtype=np.float32)
        # PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=np.float32)*(-1*np.pi)

        # PiFlip_cmplx[0:flipMin+iflip,:]=np.exp(1j*np.pi)
        PiFlip_cmplx[:,0:iflip]=np.exp(1j*np.pi)
        # PiFlip_cmplx[:,0:iflip]=(np.pi)


        # np.angle( np.random.rand(1200,1920) + np.random.rand(1200,1920) * 1j)
        ArryForSLM=slm.phaseTolevel(np.angle(PiFlip_cmplx))
        # slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
        slm.LCOS_Display(ArryForSLM, channel)
        
        
        time.sleep(slm.GLobProps[channel].RefreshTime)
        PixelFlipX=np.append(PixelFlipX,iflip)

        # Work out the coincidences
        CoinAvg=0
        countAvg=0
        for iavg in range(avgCount):
            CoinData=timeTagger.getCoincidences()# get timetagger coincidenece
            CoinAvg=CoinAvg+CoinData[2]
            countAvg=countAvg+CoinData[0]

        print(CoinAvg/avgCount)
        CountX=np.append(CountX,countAvg/avgCount)
        powerReadingX=np.append(powerReadingX,CoinAvg/avgCount)

        
        
    # top to bottom sweep    
    for iflip in range(flipMinY,flipMaxY,flipCount):
        PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)
        # PiFlip_cmplx =np.zeros((slm.slmHeigth,slm.slmWidth),dtype=np.float32)

        PiFlip_cmplx[0:iflip,:]=np.exp(1j*np.pi)
        # PiFlip_cmplx[0:flipMin+iflip,:]=(np.pi)

        # PiFlip_cmplx[:,0:flipMin+iflip]=np.exp(1j*np.pi)

        # np.angle( np.random.rand(1200,1920) + np.random.rand(1200,1920) * 1j)
        ArryForSLM=slm.phaseTolevel(np.angle(PiFlip_cmplx))
        # slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
        slm.LCOS_Display(ArryForSLM,channel)
        
        time.sleep(slm.GLobProps[channel].RefreshTime)
        PixelFlipY=np.append(PixelFlipY,iflip)
        
         # Work out the coincidences
        countAvg=0
        CoinAvg=0
        for iavg in range( avgCount):
            CoinData=timeTagger.getCoincidences()# get timetagger coincidenece
            CoinAvg=CoinAvg+CoinData[2]
            countAvg=countAvg+CoinData[0]

        print(CoinAvg/avgCount)
        CountY=np.append(CountY,countAvg/avgCount)
        powerReadingY=np.append(powerReadingY,CoinAvg/avgCount)
    
    slm.LCOS_Clean(channel)
    timeTagger.setContinuousCaptureMode()
    return powerReadingX,CountX,PixelFlipX,powerReadingY,CountY,PixelFlipY

def CourseSweepAcrossSLMPowerMeter(slm,channel,power_meter,flipCount=25):

    slm.LCOS_Clean()
    # flipMin=//2-flipCount//2
    flipMin=0
    flipMax=slm.slmHeigth//2+flipCount//2
    flipMax=slm.slmWidth//2+flipCount//2
    print(power_meter.read)
    powerReadingX=np.empty(0)
    powerReadingY=np.empty(0)

    #Left to right sweep
    for iflip in range(0,slm.slmWidth,flipCount):

        powerReadingX=np.append(powerReadingX,power_meter.read)
        # PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)
        PiFlip_cmplx =np.zeros((slm.slmHeigth,slm.slmWidth),dtype=np.float32)
        # PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=np.float32)*(-1*np.pi)

        # PiFlip_cmplx[0:flipMin+iflip,:]=np.exp(1j*np.pi)
        # PiFlip_cmplx[:,0:flipMin+iflip]=np.exp(1j*np.pi)
        PiFlip_cmplx[:,0:flipMin+iflip]=(np.pi)


        # np.angle( np.random.rand(1200,1920) + np.random.rand(1200,1920) * 1j)
        ArryForSLM=slm.phaseTolevel(np.angle(iFlip_cmplx), aperture = 1)
        # slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
        slm.LCOS_Display(ArryForSLM, channel)
        
        
        time.sleep(slm.GLobProps[channel].RefreshTime)
        
    # top to bottom sweep    
    for iflip in range(0,slm.slmHeigth,flipCount):
        powerReadingY=np.append(powerReadingY,power_meter.read)

        # PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)
        PiFlip_cmplx =np.zeros((slm.slmHeigth,slm.slmWidth),dtype=np.float32)

        # PiFlip_cmplx[0:flipMin+iflip,:]=np.exp(1j*np.pi)
        PiFlip_cmplx[0:flipMin+iflip,:]=(np.pi)

        # PiFlip_cmplx[:,0:flipMin+iflip]=np.exp(1j*np.pi)

        # np.angle( np.random.rand(1200,1920) + np.random.rand(1200,1920) * 1j)
        ArryForSLM=slm.phaseTolevel(np.angle(PiFlip_cmplx), aperture = 1)
        # slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
        slm.LCOS_Display(ArryForSLM, channel)
        
        time.sleep(slm.GLobProps[channel].RefreshTime)
    
    slm.LCOS_Clean(channel)
    return powerReadingX,powerReadingY

def ProcessFramesFromPhaseCal(FrameBuffer,MaskCount,FFTRadiusIn):
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
    pixelSize = 6.9e-6
    #Centre wavelength (nanometres)
    lambda0 = 810e-9
    #Polarisation components per frame
    polCount = 1
    #Width/height of window to FFT on the camera. (pixels)
    # nx = 256
    # ny = 256
    CamDims=FrameBuffer.shape
    frameWidth = CamDims[2]
    frameHeight = CamDims[1]
    nx = 512
    ny = 512
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
    fftWindowSizeX=512
    fftWindowSizeY=512
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
# Physically it really should matter but the SLM really hate going from 255 to 0 so makes a little bit of
# scence from that perspective. This took a week of my time as the phase cals where just absoultely terrible 
# that where coming out.
# Daniel 10min from writting this comment:
# Past Daniel is a absoulte idiot if you think about it for like 10 seconds you were doing
# the phase cal wrong. you have to start it off at 0 grey level and move it up to 255 as this is the
# whole point of the calibration. you are a idiot. I am leaving the comit here so you can feel 
# the shame every time you look at this code.
def PhaseCalibration(slm:pyLCOS.LCOS,channel,CamObj:CamForm.GeneralCameraObject,Direction):
    
    CamObj.SetSingleFrameCapMode()
    phaseLevels=256
    imask=0
    pol="V"
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
    phaseLevel_257=np.linspace(-np.pi,np.pi,257)
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
            MASK[0:int((Ny/2)),:]=np.exp(1j*phaseLevel[128])
            MASK[int((Ny/2)):Ny,:]=np.exp(1j*phaseLevel[level])
        elif(Direction=="x"):
            MASK[:,0:int((Nx/2))]=np.exp(1j*phaseLevel[128])
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
    CamObj.SetContinousFrameCapMode()


    return FrameBuffer


def CourseSweepAcrossSLM_digholo(slm:pyLCOS.LCOS,channel,CamObj:CamForm.GeneralCameraObject,flipCount,PixelsFromCenter):
    # need to set the camera to singleFrameCapturemode
    CamObj.SetSingleFrameCapMode()
    
    imask=0
    pol='V'
    PixelSize=CamObj.PixelSize
    #Centre wavelength (nanometres)
    lambda0 = 810e-9
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
    verbosity = 2
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
    fftWindowSizeX=512
    fftWindowSizeY=512
    handleIdx=digH_hpy.digHolo.digHoloCreate()

    #Clear the SLM
    slm.LCOS_Clean()
    time.sleep(slm.GLobProps[channel].RefreshTime)
    #Get a frame for to make the reference field
    CamObj.GetFrame()
    FrameBufferInitial=CamObj.FrameBuffer.astype(np.float32)
    frameBufferPtr_initial = FrameBufferInitial.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    batchCount=1
    FFTRadius=0.2
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
    # RefSigPWR_log_X=np.empty(0)
    PixelFlipX=np.empty(0)
    RefSigPWR_Y=np.empty(0)
    # RefSigPWR_log_Y=np.empty(0)
    PixelFlipY=np.empty(0)


    AvgFrameCount=10

    #Left to right sweep
    for iflip in range(flipMinX,flipMaxX,flipCount):
        PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)
        # PiFlip_cmplx =np.zeros((slm.slmHeigth,slm.slmWidth),dtype=np.float32)
        # PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=np.float32)*(-1*np.pi)

        # PiFlip_cmplx[0:flipMin+iflip,:]=np.exp(1j*np.pi)
        PiFlip_cmplx[:,0:iflip]=np.exp(1j*np.pi)
        # PiFlip_cmplx[:,0:iflip]=(np.pi)


        # np.angle( np.random.rand(1200,1920) + np.random.rand(1200,1920) * 1j)
        ArryForSLM=slm.phaseTolevel(np.angle(PiFlip_cmplx))
        slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
        time.sleep(slm.GLobProps[channel].RefreshTime)
        PixelFlipX=np.append(PixelFlipX,iflip)

        #Going to go through and grab a few frames to get a average overlap with the ref Field to make it a but more acutrate 
        overlap_avgPWR=0
        for iframe in range(AvgFrameCount):
            # print('test')
            CamObj.GetFrame()
            Frame=CamObj.FrameBuffer.astype(np.float32)
            digholoFuncWrapper.ProcessBatchOfFrames(handleIdx,1,Frame)
            Field_Sig= digholoFuncWrapper.GetField(handleIdx)
            overlap=OpticOp.overlap(np.squeeze(Field_Sig),np.squeeze(np.conj(Field_Ref)))
            # print(overlap)

            overlap_avgPWR=overlap_avgPWR+(np.sum(np.abs(overlap)**2))

    
        RefSigPWR_X=np.append(RefSigPWR_X,np.sqrt(overlap_avgPWR/AvgFrameCount))# work out the average overlap and put it in the RefSigPWR_X array


    RefSigPWR_X = np.sqrt(RefSigPWR_X)# not sure why I sqrt twice might be wrong will come back and check
    RefSigPWR_log_X = 10 * np.log10(RefSigPWR_X / RefPWR)

   
    # top to bottom sweep    
    for iflip in range(flipMinY,flipMaxY,flipCount):
        PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)
        PiFlip_cmplx[0:iflip,:]=np.exp(1j*np.pi)
        ArryForSLM=slm.phaseTolevel(np.angle(PiFlip_cmplx))
        slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
        time.sleep(slm.GLobProps[channel].RefreshTime)
        PixelFlipY=np.append(PixelFlipY,iflip)
        
          #Going to go through and grab a few frames to get a average overlap with the ref Field to make it a but more acutrate 
        overlap_avgPWR=0
        for iframe in range(AvgFrameCount):
            CamObj.GetFrame()
            Frame=CamObj.FrameBuffer.astype(np.float32)
            digholoFuncWrapper.ProcessBatchOfFrames(handleIdx,1,Frame)
            Field_Sig= digholoFuncWrapper.GetField(handleIdx)
            overlap=OpticOp.overlap(np.squeeze(Field_Sig),np.squeeze(np.conj(Field_Ref)))
            overlap_avgPWR=overlap_avgPWR+(np.sum(np.abs(overlap)**2))

    
        RefSigPWR_Y=np.append(RefSigPWR_Y,np.sqrt(overlap_avgPWR/AvgFrameCount))# work out the average overlap and put it in the RefSigPWR_X array


    RefSigPWR_Y = np.sqrt(RefSigPWR_Y)# not sure why I sqrt twice might be wrong will come back and check
    RefSigPWR_log_Y = 10 * np.log10(RefSigPWR_Y / RefPWR)
    
    slm.LCOS_Clean()

    ErrorCode=digH_hpy.digHolo.digHoloDestroy(handleIdx) 
    print(ErrorCode)
    CamObj.SetContinousFrameCapMode()
    return RefSigPWR_X,RefSigPWR_log_X,PixelFlipX,RefSigPWR_Y,RefSigPWR_log_Y,PixelFlipY


def ChangeFileForStopAliginment(StopAliginment):
    np.savez_compressed('StopAliginmentFile.npz',StopAliginment=StopAliginment)
def CheckFileForStopAliginment():
    data=np.load('StopAliginmentFile.npz')
    StopAliginment=data['StopAliginment']
    return StopAliginment

def SortVertex(vertexCount,funcVertex,xVertex):
    funcSorted=np.sort(funcVertex)
    idxVertxSorted=np.argsort(funcVertex)
    tempsortedVec=np.zeros(np.shape(xVertex))
    for ivert in range(vertexCount):
        k=idxVertxSorted[ivert]
        tempsortedVec[:,ivert]=xVertex[:,k]
    
    return funcSorted,tempsortedVec





def NelderMead(TotalDims,DimsToOpt,StepArray,xbar,vertexdims,ErrTol,maxAttempts,funcToOpt):
    attemptCount = 0

    converged = False
    vertexCount = vertexdims + 1

    xVertex =np.zeros((vertexdims, vertexCount))
    sortedxVertex=np.zeros((vertexdims, vertexCount))
    funcVertex=np.zeros(vertexCount)
    sortedfuncVertex=np.zeros(vertexCount)
    xVertexbar=np.zeros(vertexdims)
    xVertexh=np.zeros(vertexdims)
    xVertexNew=np.zeros(vertexdims)
    xVertexExp=np.zeros(vertexdims)
    xVertexOC=np.zeros(vertexdims)
    xVertexIC=np.zeros(vertexdims)
    
    #Set up the Vertices
    xVertex[:,0]=xbar
    xVertex[:,1:]=np.outer(xbar,np.ones((vertexdims,1)))+StepArray*np.eye(vertexdims)

    for vertIdx in range(vertexCount):
        if(CheckFileForStopAliginment()):
            break;
        else:
            funcVertex[vertIdx] = funcToOpt(xVertex[:,vertIdx],TotalDims,DimsToOpt)
        #eval_function2D(vertIdx, xVertex)
        
    print(funcVertex)
    print(xVertex)
    funcVertex,xVertex = SortVertex(vertexCount,funcVertex,xVertex)
    while not converged :
    	#calculate centroid of the simplex and pull the vertex that has the
        #largest value of the function
        #xbar is the centroid of the face 
        #These lines determine the mean of the all vertices except the one with the 
        # largest evalutation of the function which is the last element
        #in xVertex. The last element is in xVertex is put into xVertexh
        xVertexbar=np.mean(xVertex[:,0:((vertexCount-1)-1)],axis=1)
        xVertexh=xVertex[:,vertexCount-1];

        #calculate the next vertex and the funciton value
        xVertexNew= 2 * xVertexbar - xVertexh
        
        if(CheckFileForStopAliginment()):
            break;
        else:
            funcVertexNew=funcToOpt(xVertexNew,TotalDims,DimsToOpt)
        
        #Now we are going to determine if we should take the new point
        if (funcVertexNew < funcVertex[(vertexCount - 1) - 1]):
            if (funcVertexNew < funcVertex[0]):
                xVertexExp = 3 * xVertexbar - 2 * xVertexh
                if(CheckFileForStopAliginment()):
                    break;
                else:
                    funcVertexExp=funcToOpt(xVertexExp,TotalDims,DimsToOpt)

                if (funcVertexExp < funcVertexNew):
                    xVertex[:,(vertexCount - 1)] = xVertexExp[:]
                    if(CheckFileForStopAliginment()):
                        break;
                    else:
                        funcVertex[(vertexCount - 1)] = funcVertexExp
                    
                
                else:
                    xVertex[:,(vertexCount - 1)] = xVertexNew[:]
                    funcVertex[(vertexCount - 1)] = funcVertexNew
                
            else:
                xVertex[:,(vertexCount - 1)] = xVertexNew[:]
                funcVertex[(vertexCount - 1)] = funcVertexNew
            
        
        # if funcVertexNew > funcVertex[vertexCount-1]
        else:
            if (funcVertexNew < funcVertex[(vertexCount - 1)]):
                #calculate the outside contraction
                xVertexOC[:] = 1.5 * xVertexbar[:] - 0.5 * xVertexh[:]
                
                if(CheckFileForStopAliginment()):
                    break;
                else:
                    funcVertexOC=funcToOpt(xVertexOC,TotalDims,DimsToOpt)
                    
                #accept outside contraction
                if (funcVertexOC < funcVertexNew):
                    xVertex[:,(vertexCount - 1)] = xVertexOC[:]
                    funcVertex[(vertexCount - 1)] = funcVertexOC
                
                #otherwise shrink point toward best point this requires a reevaluation of all points
                else:
                    for vertIdx in range(1,vertexCount):
                        xVertex[:,vertIdx] = 0.5 * xVertex[:,0] + 0.5 * xVertex[:,vertIdx]
                        #Now we determine the function values for the vertices
                        #funcVertex[vertIdx] = eval_function2D(vertIdx, xVertex)
                        if(CheckFileForStopAliginment()):
                            break
                        else:
                            funcVertex[vertIdx]=funcToOpt(xVertex[:,vertIdx],TotalDims,DimsToOpt)
                    if(CheckFileForStopAliginment()):
                        break
                
            #dont really get this one but if xr is wore then previous worst we need to
            #recalulate
            else:
                #calculatuing IC point
                xVertexIC[:] = 0.5 * xVertexbar[:] + 0.5 * xVertexh[:];
                #funcVertexIC = eval_function1D(xVertexIC)
                if(CheckFileForStopAliginment()):
                    break;
                else:
                    funcVertexIC=funcToOpt(xVertexIC,TotalDims,DimsToOpt)

                if (funcVertexIC < funcVertex[(vertexCount - 1)]):
                    xVertex[:,(vertexCount - 1)] = xVertexIC[:]
                    funcVertex[(vertexCount - 1)] = funcVertexIC
                
                else:
                    for vertIdx in range(1,vertexCount):
                        xVertex[:,vertIdx] = 0.5 * xVertex[:,0] + 0.5 * xVertex[:,vertIdx]
                        #Now we determine the function values for the vertices
                        #funcVertex[vertIdx] = eval_function2D(vertIdx, xVertex)
                        if(CheckFileForStopAliginment()):
                            break
                        else:
                            funcVertex[vertIdx]=funcToOpt(xVertex[:,vertIdx],TotalDims,DimsToOpt)
                    if(CheckFileForStopAliginment()):
                        break
                
            #finished checking all the points I should put this in a function to
            #clean this up... nah fuck that.
        
        #Now going to do some checks and output some values to show how algorithm
        #is going
        #Need to calculate std of funcVertex to determine the error
        dErr = np.std(funcVertex);
        print('Function Value= ',funcVertex[0],' Error Accros Values= ',dErr, ' Verterx Value= ',xVertex[:,0])
        #if (attemptCount%10==0):
        #    print('Function Value= ',funcVertex[0],' Error Accros Values= ',dErr, ' Verterx Value= ',xVertex[:,0])
        #float dErr = Math::Abs(outBest[optimIdx] - lastErr);
        #lastErr = outBest[optimIdx];
        mean_val = np.mean(funcVertex);
        #print('mean func= ',  mean_val , ' stdev= ',  dErr);
        attemptCount=attemptCount+1
        #time.sleep(0.50)
        #This is really dump but it will check if the algoritum should stop
        #Check if the algorithum can stop
        if (dErr < ErrTol or attemptCount == maxAttempts or CheckFileForStopAliginment()):
            converged = True
            break
        
        else:
            #We now need to sort the point again for the iteration
            funcVertex,xVertex = SortVertex(vertexCount,funcVertex,xVertex)
        
    #End of while loop for convergence

    #Put the stage at the best value
    Final_optValue=funcToOpt(xVertex[:,0],TotalDims,DimsToOpt)
    
    print('Finished Optimisation in ' ,attemptCount, ' steps')
    return Final_optValue,xVertex[:,0]






# PiFlip_cmplx =np.ones((SLM_HEIGHT,SLM_WIDTH),dtype=complex)
# PiFlip_cmplx[:,:]=np.exp(1j*np.pi)
# ArryForSLM=slm.phaseTolevel(PiFlip_cmplx, aperture = 1)
# slm.LCOS_Display(ArryForSLM, ch = 0)


# #Lets get a inital field and then use that to reference back to where the beam is on the SLM
# FrameBufferInitial = camera.Get_Singleframe()

# # Fullimage ,ViewPortRGB_cam=digholoFuncWrapper.GetViewport_arr(handleIdx,FrameBufferInitial)
# # plt.imshow(Fullimage)
# Field_Ref=(digholoFuncWrapper.GetField(handleIdx))
# plt.imshow(cmplxplt.ComplexArrayToRgb(np.squeeze(Field_Ref)));
    
#     slm.LCOS_Clean()
#     timeTagger.setSingleCaptureMode()
#     timeTagger.setCountingTime(0.3*1e12)
#     avgCount=10
#     CoinAvg=0
#     for iavg in range( avgCount):
#             CoinData=timeTagger.getCoincidences()# get timetagger coincidenece
#             CoinAvg=CoinAvg+CoinData[0]
#     print(CoinAvg/avgCount)



  
#     # set up at the boundaries of the mask properties
#     x_center=slm.AllMaskProperties[channel][pol][imask].center[1]
#     y_center=slm.AllMaskProperties[channel][pol][imask].center[0]

#     print(x_center,y_center)
#     flipMinX=x_center-PixelsFromCenter
#     if flipMinX<0:
#         flipMinX=0
#     flipMaxX=x_center+PixelsFromCenter
#     if flipMaxX>slm.slmWidth:
#         flipMaxX=slm.slmWidth-1
#     flipMinY=y_center-PixelsFromCenter
#     if flipMinY<0:
#         flipMinY=0
#     flipMaxY=y_center+PixelsFromCenter
#     if flipMaxY>slm.slmHeigth:
#         flipMaxY=slm.slmHeigth-1
        
#     # flipMaxY=slm.slmHeigth//2+flipCount//2
#     # flipMax=slm.slmWidth//2+flipCount//2
    
#     powerReadingX=np.empty(0)
#     CountX=np.empty(0)
#     PixelFlipX=np.empty(0)
#     powerReadingY=np.empty(0)
#     CountY=np.empty(0)
#     PixelFlipY=np.empty(0)

#     #Left to right sweep
#     for iflip in range(flipMinX,flipMaxX,flipCount):
#         PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)
#         # PiFlip_cmplx =np.zeros((slm.slmHeigth,slm.slmWidth),dtype=np.float32)
#         # PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=np.float32)*(-1*np.pi)

#         # PiFlip_cmplx[0:flipMin+iflip,:]=np.exp(1j*np.pi)
#         PiFlip_cmplx[:,0:iflip]=np.exp(1j*np.pi)
#         # PiFlip_cmplx[:,0:iflip]=(np.pi)


#         # np.angle( np.random.rand(1200,1920) + np.random.rand(1200,1920) * 1j)
#         ArryForSLM=slm.phaseTolevel((PiFlip_cmplx))
#         slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
#         time.sleep(slm.GLobProps[channel].RefreshTime)
#         PixelFlipX=np.append(PixelFlipX,iflip)

#         # Work out the coincidences
#         CoinAvg=0
#         countAvg=0
#         for iavg in range(avgCount):
#             CoinData=timeTagger.getCoincidences()# get timetagger coincidenece
#             CoinAvg=CoinAvg+CoinData[2]
#             countAvg=countAvg+CoinData[0]

#         print(CoinAvg/avgCount)
#         CountX=np.append(CountX,countAvg/avgCount)
#         powerReadingX=np.append(powerReadingX,CoinAvg/avgCount)

        
        
#     # top to bottom sweep    
#     for iflip in range(flipMinY,flipMaxY,flipCount):
#         PiFlip_cmplx =np.ones((slm.slmHeigth,slm.slmWidth),dtype=complex)
#         # PiFlip_cmplx =np.zeros((slm.slmHeigth,slm.slmWidth),dtype=np.float32)

#         PiFlip_cmplx[0:iflip,:]=np.exp(1j*np.pi)
#         # PiFlip_cmplx[0:flipMin+iflip,:]=(np.pi)

#         # PiFlip_cmplx[:,0:flipMin+iflip]=np.exp(1j*np.pi)

#         # np.angle( np.random.rand(1200,1920) + np.random.rand(1200,1920) * 1j)
#         ArryForSLM=slm.phaseTolevel((PiFlip_cmplx))
#         slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
#         time.sleep(slm.GLobProps[channel].RefreshTime)
#         PixelFlipY=np.append(PixelFlipY,iflip)
        
#          # Work out the coincidences
#         countAvg=0
#         CoinAvg=0
#         for iavg in range( avgCount):
#             CoinData=timeTagger.getCoincidences()# get timetagger coincidenece
#             CoinAvg=CoinAvg+CoinData[2]
#             countAvg=countAvg+CoinData[0]

#         print(CoinAvg/avgCount)
#         CountY=np.append(CountY,countAvg/avgCount)
#         powerReadingY=np.append(powerReadingY,CoinAvg/avgCount)
    
#     slm.LCOS_Clean()
#     return powerReadingX,CountX,PixelFlipX,powerReadingY,CountY,PixelFlipY




# def 
# pixelSize = 6.9e-6
# #Centre wavelength (nanometres)
# lambda0 = 810e-9
# #Polarisation components per frame
# polCount = 1
# #Width/height of window to FFT on the camera. (pixels)
# # nx = 256
# # ny = 256
# CamDims=FrameBuffer.shape
# frameWidth = CamDims[1]
# frameHeight = CamDims[0]
# nx = 320
# ny = 320
# # nx = 256
# # ny = 256
# #Amount of detail to print to console. 0: Console off. 1: Basic info. 2:Debug mode. 3: You've got serious issues
# verbosity = 2
# #Sets the resolution mode of the reconstructed field.
# #0 : Full resolution. Reconstructed field will have same pixel
# #size/dimensions as the FFT window.
# #1 : Low resolution. Reconstructed field will have dimension of the IFFT
# #window. 
# resolutionMode = 0
# mplcIdx=0
# MaskCount=1
# consoleRedirectToFile=True
# maxMG=1
# handleIdx=digH_hpy.digHolo.digHoloCreate()

# # I think there is something weird going on witht he camera and I need to reset the camera object all the time.
# # I really need to fix this you are a monkey at the moment and you need to be a human at least


# #Clear the SLM
# slm.LCOS_Clean()


# PiFlip_cmplx =np.ones((SLM_HEIGHT,SLM_WIDTH),dtype=complex)
# PiFlip_cmplx[:,:]=np.exp(1j*np.pi)
# ArryForSLM=slm.phaseTolevel(PiFlip_cmplx, aperture = 1)
# slm.LCOS_Display(ArryForSLM, ch = 0)
# #Lets get a inital field and then use that to reference back to where the beam is on the SLM
# FrameBufferInitial = camera.Get_Singleframe()
# frameBufferPtr_initial = FrameBufferInitial.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
# batchCount=1
# FFTRadius=0.2
# digholoFuncWrapper.digholo_Initialise(handleIdx,batchCount,frameBufferPtr_initial,FFTRadius,pixelSize,lambda0,polCount,nx,ny,verbosity,resolutionMode,consoleRedirectToFile,frameWidth,frameHeight,maxMG)
# digholoFuncWrapper.ProcessBatchOfFrames(handleIdx,batchCount,FrameBufferInitial)
# # Fullimage ,ViewPortRGB_cam=digholoFuncWrapper.GetViewport_arr(handleIdx,FrameBufferInitial)
# # plt.imshow(Fullimage)
# Field_Ref=(digholoFuncWrapper.GetField(handleIdx))
# plt.imshow(cmplxplt.ComplexArrayToRgb(np.squeeze(Field_Ref)));
# # plt.imshow(np.abs(Field_Ref))
# FieldDims=Field_Ref.shape
# print(FieldDims)
# flipCount=15
# ResponseTime=200
# flipMin=SLM_HEIGHT//2-flipCount//2
# flipMin=650
# flipMax=SLM_HEIGHT//2+flipCount//2
# flipMax=750

# AvgFrameCount=10

# phaseCount=np.size(np.array(range(flipMin,flipMax,flipCount)))
# flipLocation=np.zeros(phaseCount,dtype=int)
# overlap_avg=np.zeros(phaseCount)
# Field_Sig=np.zeros((phaseCount,1,FieldDims[2],FieldDims[3]),dtype=type(Field_Ref[0,0,0,0]))
# iphase=0
# FrameBuffer = np.zeros((AvgFrameCount, camera.FrameHeight, camera.FrameWidth), dtype=np.float32)
# flipdir=1
# with VmbSystem.get_instance () as vmb:
#     cams = vmb.get_all_cameras ()
#     with cams [0] as cam:
#         for iflip in range(flipMin,flipMax,flipCount):
#             PiFlip_cmplx =np.ones((SLM_HEIGHT,SLM_WIDTH),dtype=complex)
#             # PiFlip_cmplx =np.zeros((SLM_HEIGHT,SLM_WIDTH),dtype=complex)
#             if(flipdir==0):
#                 PiFlip_cmplx[0:iflip,:]=np.exp(1j*np.pi)
#             else:
#                 PiFlip_cmplx[:,0:iflip]=np.exp(1j*np.pi)

#             # PiFlip_cmplx[0:flipMin+iflip,:]=np.exp(1j*np.pi)
#             ArryForSLM=slm.phaseTolevel(PiFlip_cmplx, aperture = 1)
#             slm.LCOS_Display(ArryForSLM, ch = 0)
#             time.sleep(ResponseTime*1e-3)
            
#             #Get a AvgFrameCount worth of frames
#             overlap_avgPWR=0
#             for iframe in range(AvgFrameCount):
#                 frame = cam.get_frame()
#                 Frame_int =CamForm.adjust_array_dimensions(np.squeeze( np.array(frame.as_opencv_image())))
#                 Frame=Frame_int.astype(np.float32)
#                 FrameBuffer[iframe,:,:] =Frame
#                 digholoFuncWrapper.ProcessBatchOfFrames(handleIdx,1,FrameBuffer[0:-1,:,:])
#                 Field_Sig= digholoFuncWrapper.GetField(handleIdx)
#                 # overlap=OpticOp.overlap(Field_Sig,Field_Ref)
#                 overlap=OpticOp.overlap(np.squeeze(Field_Sig),np.squeeze(Field_Ref))

#                 overlap_avgPWR=overlap_avgPWR+(np.sum(np.abs(overlap)**2))

#             # FrameBuffer=camera.Get_Singleframe()
#             # digholoFuncWrapper.ProcessBatchOfFrames(handleIdx,1,FrameBuffer)
#             # Field_Sig[iphase,:,:,:]=(digholoFuncWrapper.GetField(handleIdx))
            
#             overlap_avg[iphase] = np.sqrt(overlap_avgPWR/AvgFrameCount)
#             flipLocation[iphase] = iflip
#             iphase =iphase+1

# RefSigPWR=np.sqrt(overlap_avg)
# RefPWR=np.sum(np.abs(OpticOp.overlap(Field_Ref,Field_Ref))**2)
# RefSigPWR_log=10 * np.log10(RefSigPWR / RefPWR)
# plt.figure()
# plt.plot(flipLocation,(RefSigPWR_log))
# del handleIdx
# Idx=np.argmin(RefSigPWR_log)
# print('center',flipLocation[Idx])
