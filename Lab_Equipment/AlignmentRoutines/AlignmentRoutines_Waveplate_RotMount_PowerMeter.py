from Lab_Equipment.Config import config

# Python Libs
import numpy as np
import matplotlib.pyplot as plt
import ctypes
import copy
import time
# Defult Pploting properties 
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = [5,5]

# Power Meter Libs
import  Lab_Equipment.PowerMeter.PowerMeterObject as PMLib

# Rotation Mount Libs
import  Lab_Equipment.ElliptecThorlabMounts.ElliptecRotationMountObject as ELLRotLib

# Alginment Functions
import  Lab_Equipment.AlignmentRoutines.AlignmentFunctions as AlignFunc


import  Lab_Equipment.MyPythonLibs.SaveMaskToBinFile as SaveMaskBin


class WavePlateAngleCalibration():
    def __init__(self,PowerMeterObj:PMLib.PowerMeterObj, RotStageObj:ELLRotLib.ELL_RotationMountObj):
        # super().__init__()
        self.PowerMeterObj=PowerMeterObj
        self.RotStageObj=RotStageObj
        
        
        
    def PerformWavePlateAlignment(self,minBracket=0,maxBracket=90):#,TotalSpaceArrX,TotalSpaceArrY):
        self.minBracket=minBracket
        self.maxBracket=maxBracket
        for istage in range(self.RotStageObj.stageCount):
            OriginalAngleOffset=self.RotStageObj.AngleOffset[istage]
            self.RotStageObj.AngleOffset[istage]=0
            self.istage=istage
            self.Angles=np.empty(0)
            self.powerReading=np.empty(0)
            MinAngle,MinPwr=AlignFunc.GoldenSectionSearchContinuous(self.minBracket,self.maxBracket,self.RotStageObj.AngleResolution,self.ChangeWavePlateAngle_TakePwr)
            # Rounding the angle resolution of the mount
            MinAnglerounded=MinAngle#round(MinAngle / self.RotStageObj.AngleResolution) * self.RotStageObj.AngleResolution
            print('Final results: ',MinAnglerounded,MinAngle,MinPwr)
            print(MinAnglerounded-45)
            #NOTE the 45 is because the minimum point is when the waveplate is 45 degrees so we need to take 
            self.RotStageObj.AngleOffset[istage]=MinAnglerounded-45 #NOTE the 45 is because the minimum point is when the waveplate is 45 degrees so we need to 
            print("The new AngleOffset has been set to: " + str(self.RotStageObj.AngleOffset[istage]))
            print("orignial Angle Offset: "+ str(OriginalAngleOffset))
            print("diffence in Original and New Angle Offset: "+ str(MinAnglerounded-OriginalAngleOffset))

            
        return MinAngle,MinPwr
    
  
    def ChangeWavePlateAngle_TakePwr(self,xVal):
        if (xVal<0):
            xVal=0
        self.RotStageObj.HomeStages()
        actualAngle,_=self.RotStageObj.SetAngle(xVal,self.istage)
        # time.sleep(1e-2)
        xVal=actualAngle # set the xVal the measured Angle
        Pwr=self.PowerMeterObj.GetPower()
        self.Angles=np.append( self.Angles,xVal)
        self.powerReading=np.append( self.powerReading,Pwr)
        return xVal,Pwr
    
    def FineSweepAboutOffset(self,RotaCount=30,istage=0):
        self.RotStageObj.HomeStages()
        OriginalAngleOffset=1*self.RotStageObj.AngleOffset[istage]
        print(OriginalAngleOffset+45)
        RotaMin=(self.RotStageObj.AngleOffset[istage]+45)-RotaCount//2*self.RotStageObj.AngleResolution
        RotaMax=(self.RotStageObj.AngleOffset[istage]+45)+RotaCount//2*self.RotStageObj.AngleResolution
        if RotaMin<0:
            RotaMin=0
        if RotaMax>=360:
            RotaMax=360-self.RotStageObj.AngleResolution
        RotAngle_Sweep=np.linspace(RotaMin,RotaMax,RotaCount)
        powerReading=np.zeros(RotaCount)
        RotAngle_Actual=np.zeros(RotaCount)
        self.RotStageObj.AngleOffset[istage]=0

        for irot in range(RotaCount):
            self.RotStageObj.HomeStages()
            actualAngle,_=self.RotStageObj.SetAngle(RotAngle_Sweep[irot])
            # time.sleep(0.5)
            powerReading[irot]=self.PowerMeterObj.GetPower()
            RotAngle_Actual[irot] = actualAngle

        min_Power = np.min(powerReading)
        min_idx_Power = np.argmin(powerReading)
        minAngle_actual=RotAngle_Actual[min_idx_Power] 
        print("The Angle Offset is measured to be: " + str(minAngle_actual))
        print("orignial Angle Offset: "+ str(OriginalAngleOffset+45))
        print("diffence in Original and New Angle Offset: "+ str(minAngle_actual-(OriginalAngleOffset+45)))
        plt.figure()
        plt.plot(RotAngle_Actual,powerReading)
        self.RotStageObj.AngleOffset[istage]=OriginalAngleOffset
        return RotAngle_Actual,powerReading

    def CourseSweepWaveplatesPowerMeter(self,RotaCount=25):
        
        RotaMin=0
        RotaMax=360
        RotAngle_temp=np.linspace(RotaMin,RotaMax,RotaCount+1)
        RotAngle=np.delete(RotAngle_temp,-1) # Need to remove the last value as mount cant actually take 360 as a value 
        powerReading=np.zeros(RotaCount)
        
        #Rotate the mount and take Power Measurements
        for irot in range(RotaCount):
            powerReading[irot]=self.PowerMeterObj.GetPower()
            actualAngle,_=self.RotStageObj.SetAngle(RotAngle[irot])
            RotAngle[irot] = actualAngle
        return RotAngle,powerReading

# def CourseSweepWaveplatesPowerMeter(PowerMeterObj:PMLib.PowerMeterObj, RotStageObj:ELLRotLib.ELL_RotationMountObj,RotaCount=25):
    
#     RotaMin=0
#     RotaMax=360
#     RotAngle_temp=np.linspace(RotaMin,RotaMax,RotaCount+1)
#     RotAngle=np.delete(RotAngle_temp,-1) # Need to remove the last value as mount cant actually take 360 as a value 
#     powerReading=np.zeros(RotaCount)
    
#     #Rotate the mount and take Power Measurements
#     for irot in range(RotaCount):
#         powerReading[irot]=PowerMeterObj.GetPower()
#         actualAngle,_=RotStageObj.SetAngle(RotAngle[irot])
#         RotAngle[irot] = actualAngle
#     return RotAngle,powerReading
