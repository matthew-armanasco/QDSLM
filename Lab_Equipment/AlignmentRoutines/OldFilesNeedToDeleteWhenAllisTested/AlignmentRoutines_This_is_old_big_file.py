from Lab_Equipment.Config import config

import tomography.standard as standard
import tomography.masks as masks

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
import Lab_Equipment.digHolo.digHolo_pylibs.digholoHeader as digH_hpy  # as in header file for python... pretty clever I know (Daniel 2 seconds after writing this commment. Head slap you are a idiot )
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
# I can confirm that it worked. Essentiall you make class that has the function in it and as lone as the function only has on input and one output it will work 

class WaistAlignment:
    def __init__(self, slm, time_tagger:TimeTaggerMod, dim):
        self.slm = slm
        self.time_tagger = time_tagger
        self.dim = dim
        self.use_envelope_waist = False
        self.time_tagger.setCountingTime(0.5*1e12)

        Nx, Ny = self.slm.masksize 
        X_grid, Y_grid = np.meshgrid(
            np.arange(-Nx//2, Nx//2)*self.slm.pixel_size,
            np.arange(-Ny//2, Ny//2)*self.slm.pixel_size
        )

        self.X_grid = X_grid
        self.Y_grid = Y_grid
        
        self.phase_masks = np.zeros(
            (self.slm.modeCount, self.slm.MaskCount, 
             self.slm.masksize[0], self.slm.masksize[1]), 
             dtype=complex
        )

    def performWaistSweep(self, func_to_run, waist_bracket, N_points):
        waists = np.linspace(waist_bracket[0], waist_bracket[1], N_points)
        
        SNRs = np.zeros(N_points)
        coincidence_traces = np.zeros(N_points)

        for i in range(N_points):
            print("Iteration:", i)
            _, SNR, coincidence_trace = func_to_run(waists[i])
            SNRs[i] = SNR
            coincidence_traces[i] = coincidence_trace
        
        return waists, SNRs, coincidence_traces

    def performWaistAlignment(self, func_to_minimise, waist_bracket: tuple, waist_tol=5e-6):               
        optimal_waist = GoldenSectionSearchContinuous(
            waist_bracket[0], 
            waist_bracket[1],
            waist_tol,
            func_to_minimise
        )

        return optimal_waist

    # def generateMasksForWaist(self, waist):
    #     computational_basis = masks.get_computational_basis(self.X_grid, self.Y_grid, self.dim, 
    #                                                        amp_mod=True, waist=waist)

    #     waist_output_field = 288e-6

    #     # output_field = fftshift(fft2(ifftshift(tomo.get_LG_mode(0,0,waist_output_field,self.X_grid,self.Y_grid))))
    #     output_field = fftshift(fft2(ifftshift(tomo.get_LG_mode(0,0,waist_output_field,self.X_grid,self.Y_grid))))
        
    #     aperture = aperture = tomo.get_aperture(self.X_grid, self.Y_grid, 2*waist_output_field, aperture_blur=0)

    #     for i in range(self.dim):
    #         phase_mask,_,mode_power,_,mode_quality = tomo.gerchberg_saxton(
    #             computational_basis[:,:,i],
    #             output_field,
    #             20,
    #             aperture,
    #             portion_max_power=1,
    #             pixel_size=self.slm.pixel_size
    #         )

    #         print(mode_power, mode_quality)
    #         self.phase_masks[i+9,0,:,:] = np.conj(phase_mask)
        
    #     return self.phase_masks

    def getSNRForWaist(self, waist):
        computational_basis = masks.get_computational_basis(self.X_grid, self.Y_grid, self.dim, amp_mod=True, waist=waist)
        
        phase_masks,mode_powers,mode_qualities = masks.get_gs_masks_for_fields(computational_basis, self.X_grid, 
                                                                               self.Y_grid, pixel_size=self.slm.pixel_size, N_iterations=15)
        self.phase_masks[9:, :, :, :] = phase_masks

        print("mode_powers", mode_powers)
        print("mode_qualities", mode_qualities)

        # print("Getting coincidences")
        self.slm.setMaskArray("Red", self.phase_masks)
        self.slm.setMaskArray("Green", self.phase_masks)

        coincidences = np.zeros((self.dim, self.dim))

        for red_index in range(self.dim ):
            self.slm.setmask("Red", red_index + 9)

            for green_index in range(self.dim):
                self.slm.setmask("Green", green_index + 9)
                counts_data = self.time_tagger.getCoincidences()
                coincidences[red_index, green_index] = counts_data[2]

        print("coincidences", coincidences)

        SNR = np.trace(coincidences)/(np.sum(coincidences) - np.trace(coincidences))
        coincidence_trace = np.trace(coincidences)

        return waist, -SNR, coincidence_trace
    
    def getCoincidencesForWaist(self, waist):
        phase_masks = self.generateMasksForWaist(waist)

        # print("Getting coincidences")
        self.slm.setMaskArray("Red", phase_masks)
        self.slm.setMaskArray("Green", phase_masks)

        self.slm.setmask("Red", 11)
        self.slm.setmask("Green", 11)

        # counts_data = np.zeros((4, self.slm.modeCount))

        # mode_counter = 0
        # for red_index in range(self.slm.modeCount):
        #     self.slm.setmask("Red", red_index)

        #     for green_index in range(self.slm.modeCount):
        #         self.slm.setmask("Green", green_index)
        #         counts_data[:, mode_counter] = self.time_tagger.getCoincidences()
        #         mode_counter += 1

        # _,fidelity = tomo.get_fidelity_from_data(counts_data)

        counts = self.time_tagger.getCoincidences(True)

        # print("Finished getting coincidences")

        return (waist, -1*counts[2])

class CenterAlginmentSpaceCoindences():
    def __init__(self,slmObject:pyLCOS.LCOS,slmChannel ,TtaggerObj:TimeTaggerMod,DescreteSpaceArrayX,DescreteSpaceArrayY):
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
        self.DescreteSpaceArrayX=DescreteSpaceArrayX
        self.DescreteSpaceArrayY=DescreteSpaceArrayY
        
        
        #set the time tagger to single capture
        self.timeTagger.setSingleCaptureMode()
        self.timeTagger.setCountingTime(0.1*1e12)
        self.avgCount=10
        
    def PerformCenterAlignment(self):#,TotalSpaceArrX,TotalSpaceArrY):
        self.timeTagger.setSingleCaptureMode()
        self.flipdir=0 # flipdir X
        #Need to do 2 flips in the same direction for a better center one has the flip reversed
        self.Phasedir=0
        minValX_1,minIdxX_1=GoldenSelectionSearch(self.DescreteSpaceArrayX[0],self.DescreteSpaceArrayX[-1],1,self.ChangePiFlipTakeCoincidence)
        
        MinXCenter_1=minIdxX_1#self.DescreteSpaceArrayX[int(minIdxX_1)]
        # reversed flip in same direction
        self.Phasedir=1
        minValX_2,minIdxX_2=GoldenSelectionSearch(self.DescreteSpaceArrayX[0],self.DescreteSpaceArrayX[-1],1,self.ChangePiFlipTakeCoincidence)
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
        minValY_1,minIdxY_1=GoldenSelectionSearch(self.DescreteSpaceArrayY[0],self.DescreteSpaceArrayY[-1],1,self.ChangePiFlipTakeCoincidence)
        # MinYCenter_1=self.DescreteSpaceArrayY[int(minIdxY_1)]
        MinYCenter_1=minIdxY_1
        
        # reversed flip in same direction
        self.Phasedir=1
        minValY_2,minIdxY_2=GoldenSelectionSearch(self.DescreteSpaceArrayY[0],self.DescreteSpaceArrayY[-1],1,self.ChangePiFlipTakeCoincidence)
        # MinYCenter_2=self.DescreteSpaceArrayY[int(minIdxY_2)]
        MinYCenter_2=minIdxY_2
        
        # Take the average of the 2 centers
        MinYCenter=(MinYCenter_1+MinYCenter_2)/2.0
        print("min y center values")
        print(MinYCenter_1,MinYCenter_2)
        self.slm.LCOS_Clean(self.channel)
        self.timeTagger.setContinuousCaptureMode()
     

        return int(MinXCenter),int(MinYCenter)
    
  
    def ChangePiFlipTakeCoincidence(self,xVal):
        if(self.flipdir==0):
            xVal,x1Idx=CovertCont2Desc(xVal,self.DescreteSpaceArrayX);
        else:
            xVal,x1Idx=CovertCont2Desc(xVal,self.DescreteSpaceArrayY);
            
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
            CoinData=self.timeTagger.getCoincidences(True)# get timetagger coincidenece
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
       
        return xVal,CoinAvg/self.avgCount
        


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
def GoldenSelectionSearch(bracketVal_a,bracketVal_b,dspace_Tol,FuncToMinamise):
    # xArr= TotalSpaceArr#= FieldSuperPixel(FieldToProb, PixelIdxRange,PhaseValue)
    iNumIterations=0
    goldenRation=(np.sqrt(5)-1)/2;
    
    # Lets just do a course check on the problem space and see if we can get close to the root to reduce the number of calls to the function
    #aIdx,bIdx=CourseSearchOptimisation(aIdx,bIdx,ProblemProps,MinFunc)
    
    # a=xArr[aIdx]
    # b=xArr[bIdx]
    a=bracketVal_a
    b=bracketVal_b
    x1 = a+(1-goldenRation)*(b-a);
    # x1,x1Idx=CovertCont2Desc(x1,xArr);
    x2 = a+goldenRation*(b-a);
    # x2,x2Idx=CovertCont2Desc(x2,xArr);
    
    x1,f1=FuncToMinamise(x1);
    x2,f2=FuncToMinamise(x2);
    print(f1,f2)
    # dspace = np.abs(x1Idx-x2Idx);
    dspace = np.abs(x1 - x2)
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
            # x1,x1Idx=CovertCont2Desc(x1,xArr);
            f2=f1; 
            x1,f1=FuncToMinamise(x1);  # single function evaluation
        else: #otherwise, replace a with x1
            a=x1; 
            x1=x2 
            x2 = a+goldenRation*(b-a);
            # x2,x2Idx=CovertCont2Desc(x2,xArr);
            f1=f2; 
            x2,f2=FuncToMinamise(x2);  # single function evaluation
        
        dspace = abs(x2-x1)
        print(f1,x1,dspace)
        iNumIterations=iNumIterations+1 
          
    #Work out which one is the best value to take. This is kind of doing a y=(a+b)/2 but since we are dealing with integer we are not doing a half thing we are 
    #just working out which one is a the lowest
    # a,aIdx=CovertCont2Desc(a,xArr);
    # b,bIdx=CovertCont2Desc(b,xArr);
    a,f_aAndb[0]=FuncToMinamise(a)
    b,f_aAndb[1]=FuncToMinamise(b)
    f_aAndbIdx[0]=a
    f_aAndbIdx[1]=b
    minIdx=np.argmin(f_aAndb)
    minValue=np.min(f_aAndb)
    #print(iNumIterations)
    
    return -1*minValue,f_aAndbIdx[minIdx]

def GoldenSectionSearchContinuous(bracketVal_a,bracketVal_b,dspace_Tol,FuncToMinamise):
    iNumIterations = 0
    goldenRation=(np.sqrt(5)-1)/2

    a=bracketVal_a
    b=bracketVal_b
    x1 = a+(1-goldenRation)*(b-a)
    x2 = a+goldenRation*(b-a)
    
    x1,f1=FuncToMinamise(x1)
    x2,f2=FuncToMinamise(x2)
    print(x1,x2)
    print(f1,f2)

    dspace = np.abs(x1 - x2)

    while(dspace > dspace_Tol):
        print("Iteration:", iNumIterations)
        if (f1 < f2): # if f(x1) < f(x2), replace b with x2
            b=x2; 
            x2=x1; 
            x1=a+(1-goldenRation)*(b-a);

            f2=f1; 
            x1,f1=FuncToMinamise(x1);  # single function evaluation
        else: #otherwise, replace a with x1
            a=x1; 
            x1=x2 
            x2 = a+goldenRation*(b-a);

            f1=f2; 
            x2,f2=FuncToMinamise(x2);  # single function evaluation
        
        dspace = abs(x2-x1)
        print((x1,x2), (f1,f2), dspace)
        iNumIterations=iNumIterations+1 

    _,f_a = FuncToMinamise(a)
    _,f_b = FuncToMinamise(b)
    avg_x = (a+b)/2
    avg_f = (f_a+f_b)/2
    
    return avg_x, avg_f

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
            CoinData=timeTagger.getCoincidences(True)# get timetagger coincidenece
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
            CoinData=timeTagger.getCoincidences(True)# get timetagger coincidenece
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
            CoinData=timeTagger.getCoincidences(True)# get timetagger coincidenece
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
            CoinData=timeTagger.getCoincidences(True)# get timetagger coincidenece
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
            CoinData=timeTagger.getCoincidences(True)# get timetagger coincidenece
            CoinAvg=CoinAvg+CoinData[2]
            countAvg=countAvg+CoinData[0]

        print(CoinAvg/avgCount)
        CountY=np.append(CountY,countAvg/avgCount)
        powerReadingY=np.append(powerReadingY,CoinAvg/avgCount)
    
    slm.LCOS_Clean(channel)
    timeTagger.setContinuousCaptureMode()
    return powerReadingX,CountX,PixelFlipX,powerReadingY,CountY,PixelFlipY


def CourseSweepAcrossSLM_digholo(slm:pyLCOS.LCOS,channel,CamObj:CamForm.GeneralCameraObject,flipCount,PixelsFromCenter):
     # need to set the camera to singleFrameCapturemode
    CamObj.Exposure
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
        # slm.LCOS_Display(ArryForSLM, slm.GLobProps[channel].rgbChannelIdx)
        slm.LCOS_Display(ArryForSLM, channel)
        
        
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
    
    slm.LCOS_Clean(channel)

    ErrorCode=digH_hpy.digHolo.digHoloDestroy(handleIdx) 
    print(ErrorCode)
    CamObj.SetContinousFrameCapMode(CamObj.Exposure.value)
    return RefSigPWR_X,RefSigPWR_log_X,PixelFlipX,RefSigPWR_Y,RefSigPWR_log_Y,PixelFlipY



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


def CourseSweepAcrossSLM_digholo_old(slm:pyLCOS.LCOS,channel,CamObj:CamForm.GeneralCameraObject,flipCount,PixelsFromCenter):
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


class MultiDimAlginmentSpace():
    def __init__(self,slmObject:pyLCOS.LCOS,TtaggerObj:TimeTaggerMod,TotalDims,DimToOpt,imask=0,pol='V',imode=0):
        super().__init__()
        self.slm=slmObject
        self.timeTagger=TtaggerObj
        self.imask=int(imask)
        self.pol=pol
        self.imode=int(imode)
        #set the time tagger to single capture
        self.timeTagger.setSingleCaptureMode()
        # self.timeTagger.setCountingTime(0.4*1e12)
        self.timeTagger.setCountingTime( 0.05*1e12 )

        self.avgCount=20
        self.TotalDims=TotalDims
        self.DimToOpt=DimToOpt
        self.modeCount=slmObject.modeCount
    
    def UpdateSLMProp(self,channel,pol,imask,updateVal,PropToUpdate):
        if(PropToUpdate==0):
            self.slm.AllMaskProperties[channel][pol][imask].zernike.zern_coefs[0]=updateVal
        elif(PropToUpdate==1):
            self.slm.AllMaskProperties[channel][pol][imask].center[0]=int(updateVal)
            updateVal=int(updateVal)
        elif(PropToUpdate==2):
            self.slm.AllMaskProperties[channel][pol][imask].center[1]=int(updateVal)
            updateVal=int(updateVal)
        elif(PropToUpdate==3):
            self.slm.AllMaskProperties[channel][pol][imask].zernike.zern_coefs[1]=updateVal
        elif(PropToUpdate==4):
            self.slm.AllMaskProperties[channel][pol][imask].zernike.zern_coefs[2]=updateVal
        elif(PropToUpdate==5):
            self.slm.AllMaskProperties[channel][pol][imask].zernike.zern_coefs[4]=updateVal
        else:
            print("major problem. This varible is not implemented in the multidim algorithm")
        return updateVal
    
    def UpdateVertex_GetSNR(self,xVertexSingle):
        #Move all the allowed Dimensions
        vertDimCount=0
        idimRed=0
        idimGreen=0
        #Update all the SLM properties 
        for idim in range(self.TotalDims):
            if (idim<(self.TotalDims/2.0)):#FibreFirst
                if(self.DimToOpt[idim]==True  ):
                    xVertexSingle[vertDimCount]=self.UpdateSLMProp("Red",self.pol,self.imask,xVertexSingle[vertDimCount],idimRed)
                    vertDimCount=vertDimCount+1
                idimRed=idimRed+1
                    
            else:
                if(self.DimToOpt[idim]==True):
                    xVertexSingle[vertDimCount]=self.UpdateSLMProp("Green",self.pol,self.imask,xVertexSingle[vertDimCount],idimGreen)
                    vertDimCount=vertDimCount+1
                idimGreen=idimGreen+1

        # counts_data = np.zeros((4, 1,self.modeCount,self.modeCount))
        # mode_counter=0
        # for imode in range(self.modeCount):
        #     self.slm.setmask("Red", imode)
        #     for jmode in range(self.modeCount):
        #         self.slm.setmask("Green", jmode)
        #         counts_data[:,0, imode,jmode] = self.timeTagger.getCoincidences()
        #         mode_counter += 1       
        mode_counter=0
        modeCenter=20
        shiftmode=5
        modeStart=modeCenter-shiftmode
        modeEnd=modeCenter+shiftmode+1
        modeNum=modeEnd-modeStart

        # modestsrt=9
        # modeend=11
        # modeNum=(modeend+1)-modestsrt
        counts_data = np.zeros((4,modeNum, modeNum))
        mode_counter=0
        i=0
        for imode in range(modeStart,modeEnd):
            self.slm.setmask("Red", imode)
            j=0
            for jmode in range(modeStart,modeEnd):
                self.slm.setmask("Green", jmode)
                counts_data[:,i,j] = self.timeTagger.getCoincidences()
                j=j+1
                mode_counter += 1 
            i=i+1
        # for imode in range(modestsrt,modeend+1):
        #     self.slm.setmask("Red", imode)
        #     for jmode in range(9,12):
        #         self.slm.setmask("Green", jmode)
        #         counts_data[imode,jmode] = self.timeTagger.getCoincidences()
        #         mode_counter += 1  
                # Apply the updated SLM properties
        # for channel in self.slm.ActiveRGBChannels:
        #     self.slm.setmask(channel,self.imode)
        
            
        # CoinAvg=0
        # countAvg=0
        # for iavg in range(self.avgCount):
        #     CoinData=self.timeTagger.getCoincidences(True)# get timetagger coincidenece
        #     CoinAvg=CoinAvg+CoinData[2]
        #     countAvg=countAvg+CoinData[0]

            #get the feds
        SNR,DiagPwrSum,_,_,_ = MetricCals.CalculateSNR(counts_data[2,:,:])
        # SNR,DiagPwrSum,Visibility,DiagCoup,OffDiagCoup
        # _,fed = standard.get_fidelity_from_data(counts_data)

        # print(CoinAvg/avgCount)
        # MetricVaule=-1*DiagPwrSum
        MetricVaule=-1*SNR

        
        return MetricVaule,xVertexSingle
    
    def UpdateVertex_GetCoincidences(self,xVertexSingle):
           #Move all the allowed Dimensions
        vertDimCount=0
        idimRed=0
        idimGreen=0
        #Update all the SLM properties 
        for idim in range(self.TotalDims):
            if (idim<(self.TotalDims/2.0)):#FibreFirst
                if(self.DimToOpt[idim]==True  ):
                    xVertexSingle[vertDimCount]=self.UpdateSLMProp("Red",self.pol,self.imask,xVertexSingle[vertDimCount],idimRed)
                    vertDimCount=vertDimCount+1
                idimRed=idimRed+1
                    
            else:
                if(self.DimToOpt[idim]==True):
                    xVertexSingle[vertDimCount]=self.UpdateSLMProp("Green",self.pol,self.imask,xVertexSingle[vertDimCount],idimGreen)
                    vertDimCount=vertDimCount+1
                idimGreen=idimGreen+1
                
        # Apply the updated SLM properties
        for channel in self.slm.ActiveRGBChannels:
            self.slm.setmask(channel,self.imode)
            
        CoinAvg=0
        countAvg=0
        for iavg in range(self.avgCount):
            CoinData=self.timeTagger.getCoincidences(True)# get timetagger coincidenece
            CoinAvg=CoinAvg+CoinData[2]
            countAvg=countAvg+CoinData[0]

        # print(CoinAvg/avgCount)
        MetricVaule=-1*CoinAvg/self.avgCount
        
        return MetricVaule,xVertexSingle
    

    def UpdateVertex_GetFed(self,xVertexSingle):
           #Move all the allowed Dimensions
        vertDimCount=0
        idimRed=0
        idimGreen=0
        #Update all the SLM properties 
        for idim in range(self.TotalDims):
            if (idim<(self.TotalDims/2.0)):#FibreFirst
                if(self.DimToOpt[idim]==True  ):
                    xVertexSingle[vertDimCount]=self.UpdateSLMProp("Red",self.pol,self.imask,xVertexSingle[vertDimCount],idimRed)
                    vertDimCount=vertDimCount+1
                idimRed=idimRed+1
                    
            else:
                if(self.DimToOpt[idim]==True):
                    xVertexSingle[vertDimCount]=self.UpdateSLMProp("Green",self.pol,self.imask,xVertexSingle[vertDimCount],idimGreen)
                    vertDimCount=vertDimCount+1
                idimGreen=idimGreen+1

        # counts_data = np.zeros((4, 1,self.modeCount,self.modeCount))
        # mode_counter=0
        # for imode in range(self.modeCount):
        #     self.slm.setmask("Red", imode)
        #     for jmode in range(self.modeCount):
        #         self.slm.setmask("Green", jmode)
        #         counts_data[:,0, imode,jmode] = self.timeTagger.getCoincidences()
        #         mode_counter += 1       

        counts_data = np.zeros((4, self.modeCount**2))
        mode_counter=0
        for imode in range(self.modeCount):
            self.slm.setmask("Red", imode)
            for jmode in range(self.modeCount):
                self.slm.setmask("Green", jmode)
                counts_data[:, mode_counter] = self.timeTagger.getCoincidences()
                mode_counter += 1  
                # Apply the updated SLM properties
        # for channel in self.slm.ActiveRGBChannels:
        #     self.slm.setmask(channel,self.imode)
            
        # CoinAvg=0
        # countAvg=0
        # for iavg in range(self.avgCount):
        #     CoinData=self.timeTagger.getCoincidences(True)# get timetagger coincidenece
        #     CoinAvg=CoinAvg+CoinData[2]
        #     countAvg=countAvg+CoinData[0]

            #get the feds

        _,fed = standard.get_fidelity_from_data(counts_data)

        # print(CoinAvg/avgCount)
        MetricVaule=-1*fed
        
        return MetricVaule,xVertexSingle
        
        



def GetSLMSetting(slm:pyLCOS.LCOS,Channel,pol="V",imask=0):
    VertexArr=np.empty(0)
    VertexArr=np.append(VertexArr,slm.AllMaskProperties[Channel][pol][imask].zernike.zern_coefs[0])
    VertexArr=np.append(VertexArr,slm.AllMaskProperties[Channel][pol][imask].center[0])
    VertexArr=np.append(VertexArr,slm.AllMaskProperties[Channel][pol][imask].center[1])
    VertexArr=np.append(VertexArr,slm.AllMaskProperties[Channel][pol][imask].zernike.zern_coefs[1])
    VertexArr=np.append(VertexArr,slm.AllMaskProperties[Channel][pol][imask].zernike.zern_coefs[2])
    VertexArr=np.append(VertexArr,slm.AllMaskProperties[Channel][pol][imask].zernike.zern_coefs[4])
    
    return VertexArr

def ChangeFileForStopAliginment(StopAliginment):
    # np.savez_compressed(config.MPLC_LIB_PATH+'StopAliginmentFile.npz',StopAliginment=StopAliginment)
    with open(config.MPLC_LIB_PATH+'StopAliginmentFile.txt', 'w') as file:
        file.write(str(StopAliginment))  # Write a string to the file
def CheckFileForStopAliginment():
    # data=np.load(config.MPLC_LIB_PATH+'StopAliginmentFile.npz')
    # StopAliginment=data['StopAliginment']
    with open(config.MPLC_LIB_PATH+'StopAliginmentFile.txt', 'r') as file:
        StopAliginment = file.read()  # Read the entire file content
    return int(StopAliginment)

def SortVertex(vertexCount,funcVertex,xVertex):
    funcSorted=np.sort(funcVertex)
    idxVertxSorted=np.argsort(funcVertex)
    tempsortedVec=np.zeros(np.shape(xVertex))
    for ivert in range(vertexCount):
        k=idxVertxSorted[ivert]
        tempsortedVec[:,ivert]=xVertex[:,k]
    
    return funcSorted,tempsortedVec





def NelderMead(StepArray,xbar,vertexdims,ErrTol,maxAttempts,funcToOpt):
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
            funcVertex[vertIdx],xVertex[:,vertIdx] = funcToOpt(xVertex[:,vertIdx])
        #eval_function2D(vertIdx, xVertex)
        
    print(funcVertex)
    print(xVertex)
    funcVertex,xVertex = SortVertex(vertexCount,funcVertex,xVertex)
    itime=0
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
            funcVertexNew,xVertexNew=funcToOpt(xVertexNew)
        
        #Now we are going to determine if we should take the new point
        if (funcVertexNew < funcVertex[(vertexCount - 1) - 1]):
            if (funcVertexNew < funcVertex[0]):
                xVertexExp = 3 * xVertexbar - 2 * xVertexh
                if(CheckFileForStopAliginment()):
                    break;
                else:
                    funcVertexExp,xVertexExp=funcToOpt(xVertexExp)

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
                    funcVertexOC,xVertexOC=funcToOpt(xVertexOC)
                    
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
                            funcVertex[vertIdx],xVertex[:,vertIdx]=funcToOpt(xVertex[:,vertIdx])
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
                    funcVertexIC,xVertexIC=funcToOpt(xVertexIC)

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
                            funcVertex[vertIdx],xVertex[:,vertIdx]=funcToOpt(xVertex[:,vertIdx])
                    if(CheckFileForStopAliginment()):
                        break
                
            #finished checking all the points I should put this in a function to
            #clean this up... nah fuck that.
        
        #Now going to do some checks and output some values to show how algorithm
        #is going
        #Need to calculate std of funcVertex to determine the error
        dErr = np.std(funcVertex);
        print(attemptCount,' Function Value= ',funcVertex[0],' Error Accros Values= ',dErr, ' Verterx Value= ',xVertex[:,0])
        print(funcVertex[:])
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
    Final_optValue=funcToOpt(xVertex[:,0])
    
    print('Finished Optimisation in ' ,attemptCount, ' steps')
    return Final_optValue,xVertex[:,0]





