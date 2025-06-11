# import Lab_Equipment.Config.config as config
from Lab_Equipment.Config import config 
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import shared_memory
import copy
import cv2
import time
import ctypes
class GeneralCameraObject():
        def __init__(self,CameraType,CalibrationFilename,Nx,Ny,FrameWidth,FrameHeight,FrameDim,Framedtype,FrameBuffer,PixelSize,Exposure,Offset=0,Gain=0,FanState=0,CaptureMode=0,CameraName=''):
            super().__init__()
            self.CameraType=CameraType
            self.Ny = Ny
            self.Nx = Nx       
            print(self.Ny,self.Nx )
            self.FrameBuffer = FrameBuffer 
            # self.FrameBuffer =adjust_array_dimensions(np.squeeze( np.zeros((self.Ny,self.Nx),dtype=np.uint8)))
            self.FrameDim=FrameDim
            self.FrameHeight = FrameHeight
            self.FrameWidth = FrameWidth
            self.Framedtype=Framedtype
            self.PixelSize=PixelSize
            self.CameraName=CameraName
  
            # NOTE we need to make a shared memory space for the camera frames for user to access them.
            # This seems to be the most memory efficenct way to do it.   
            # self.shm = shared_memory.SharedMemory(create=True, size=int(np.prod((self.FrameHeight, self.FrameWidth)) * np.dtype(np.uint8).itemsize))
            self.shm = shared_memory.SharedMemory(create=True, size=int(self.FrameHeight* self.FrameWidth * np.dtype(self.Framedtype).itemsize))
            self.FrameBuffer_SharedMem  = np.ndarray((self.FrameHeight, self.FrameWidth), dtype=self.FrameBuffer.dtype, buffer=self.shm.buf)                    

            self.shm_digholo = shared_memory.SharedMemory(create=True, size=int(self.FrameHeight* self.FrameWidth * np.dtype(self.Framedtype).itemsize))
            self.FrameBuffer_digholo_SharedMem  = np.ndarray((self.FrameHeight, self.FrameWidth), dtype=self.FrameBuffer.dtype, buffer=self.shm_digholo.buf)   
            
            #We need to set up a bunch of varibles for the threading of the camera.   
            self.terminateCamera = multiprocessing.Event()
            self.GetFrameFlag = multiprocessing.Event()
            self.GetFrameFlag_digholo = multiprocessing.Event()
            self.ContinuesMode= multiprocessing.Event()
            self.SingleFrameMode= multiprocessing.Event()
            self.SetCamCalibrationEvent= multiprocessing.Event()
            
            #Fan
            self.SetFanFlag = multiprocessing.Event()
            self.FanState=FanState

            # Exposure 
            self.SetExposureFlag = multiprocessing.Event()
            self.ExposureSet=multiprocessing.Value('i', 0)
            self.Exposure=Exposure
            
            # Gain
            self.SetGainFlag = multiprocessing.Event()
            self.GainSet=multiprocessing.Value('i', 0)
            self.Gain=Gain
            
            # Offset not always the same for all cameras
            self.SetOffsetFlag = multiprocessing.Event()
            self.OffsetSet=multiprocessing.Value('i', 0)
            self.Offset=Offset
            

            #capture mode is something just the NiT and I think xenics cameras 
            self.SetCaptureModeFlag = multiprocessing.Event()
            self.CaptureModeSet=multiprocessing.Value('i', 0)
            self.CaptureMode=CaptureMode
            

            #If you want to look at the log of the camera in the live window you can set this flag currently it is only implemented for QImagCam
            self.LogPlot= multiprocessing.Event()

            self.FrameObtained = multiprocessing.Value('i', 0)  # Flag to indicate when a frame is ready  
            self.FrameHeightThread=multiprocessing.Value('i', self.FrameHeight)
            self.FrameWidthThread=multiprocessing.Value('i', self.FrameWidth)
            self.frame_queue = multiprocessing.Queue()
            
            # Shared varibles to communicate camera properites
            self.shared_float = multiprocessing.Value("f", 0)
            self.shared_int = multiprocessing.Value('i', 0)
            self.shared_flag_int = multiprocessing.Value('i', 0)
            
            manager = multiprocessing.Manager() # this lets us share a dictionary between threads
            self.Cam_Calibtation = manager.dict(
                { "CalibrationFilename": CalibrationFilename,
            })
        
            
            
        def GetFrame(self,ConvertToFloat32=False):  
            self.GetFrameFlag.set()
            while self.FrameObtained.value == 0:
                time.sleep(1e-9)
                
            #Pull the frame from the shared buffer and save it to the camera object buffer so an other process can get to it.
            # self.FrameBuffer= np.array(self.FrameBuffer_SharedMem)  # Make a copy of the frame
            np.copyto( self.FrameBuffer,self.FrameBuffer_SharedMem)

            self.FrameObtained.value = 0
            if (ConvertToFloat32):
                return self.FrameBuffer.astype(np.float32)
            else:
                return  self.FrameBuffer          
            # #try and grab a frame until you get one.
            # while self.FrameObtained.value == 0:
            #     time.sleep(1e-9)
            #     # self.GetFrameFlag.set()
            # #Pull the frame from the shared buffer and save it to the camera object buffer so an other process can get to it.
            # self.Framebuffer = np.array(self.FrameBuffer_SharedMem)  # Make a copy of the frame
            # self.FrameObtained.value = 0
            # return self.Framebuffer
            #NOTE the code below can be used if you wanted to go the route of using the queue instead of sharded memory
            # self.GetFrameFlag.set()
            # while self.frame_queue.empty() == True:
            #     self.GetFrameFlag.set()
            # frame_bytes = self.frame_queue.get_nowait()
            # # self.frame_queue.empty()
            # self.Framebuffer = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(self.FrameDim)
            # self.FrameObtained.value = 0
        
        def SaveFrameToFile(self,FileName,Frame):
            np.save('Data\\SavedFrame_'+FileName+'.npy', Frame)

        def LoadFrameFromFile(self,FileName):
            frameLoaded=np.load('Data\\SavedFrame_'+FileName+'.npy')
            return frameLoaded
       
        def SetExposure(self,NewExposure): 
            # self.Exposure.value=NewExposure
            self.shared_float.value=NewExposure
            self.SetExposureFlag.set()
            while self.SetExposureFlag.is_set():
                time.sleep(1e-12)
            self.shared_flag_int.value=0
            self.Exposure=self.shared_float.value
            print('New Exposure time (us)= ',self.Exposure*1e-6)
            return

        def SetGain(self,NewGain): 
            self.shared_float.value=NewGain
            self.SetGainFlag.set()
            while self.SetGainFlag.is_set():
                time.sleep(1e-12)
            self.shared_flag_int.value=0
            self.Gain=self.shared_float.value
            print('New Gain = ',self.Gain)

        def SetFanState(self,NewFanState): 
            self.shared_float.value=NewFanState
            self.SetFanFlag.set()
            while self.SetFanFlag.is_set():
                time.sleep(1e-12)
            self.shared_flag_int.value=0
            self.FanState=self.shared_float.value
            print('New Gain = ',self.FanState)

        def SetOffset(self,NewOffset): 
            self.shared_float.value=NewOffset
            while self.OffsetSet.is_set():
                time.sleep(1e-12)
                # self.SetOffsetFlag.set()
            self.shared_flag_int.value=0
            self.Offset=self.shared_float.value
            print('New Offset= ',self.Offset)
            
        def SetCalibrationFile(self,NewCalibrationFilename): 
            CalibrationFile=config.CAMERA_LIB_PATH+"CameraCalibrations\\"+self.CameraType+"\\"+ NewCalibrationFilename
            self.Cam_Calibtation["CalibrationFilename"]=CalibrationFile
            print(CalibrationFile)
            
            self.SetCamCalibrationEvent.set()
            while self.SetCamCalibrationEvent.is_set():
                time.sleep(1e-12)
            print(self.shared_int.value)
            
        
        def SetLogPlot(self):
            if (self.LogPlot.is_set()):
                self.LogPlot.clear()
            else:
                self.LogPlot.set()

        def SetSingleFrameCapMode(self):
            self.ContinuesMode.clear()
            self.SingleFrameMode.set()
        def SetContinousFrameCapMode(self,Exposure=None):
            if Exposure is None:
                Exposure=self.Exposure
            self.SetExposure(Exposure)
            self.SingleFrameMode.clear()
            self.ContinuesMode.set()
            


def PlotFrames(iframe,Framebuffer):
        fig, ax1=plt.subplots();
        # fig.subplots_adjust(wspace=0.1, hspace=-0.6);
        # ax1.cmplxplt.complexColormap(frame);
        ax1.imshow(np.squeeze(Framebuffer[iframe,:,:]),cmap='gray');
        ax1.set_title('CameraFrames',fontsize = 8);
        ax1.axis('off'); 
        
def adjust_array_dimensions(arr):
            # Get the current dimensions of the array
            rows, cols = arr.shape

            # Calculate the number of rows and columns to add on each side
            rows_to_add = (16 - (rows % 16)) % 16
            cols_to_add = (16 - (cols % 16)) % 16

            # Calculate the new dimensions
            new_rows = rows + rows_to_add
            new_cols = cols + cols_to_add
            # Check if adjustment is needed
            if new_rows != rows or new_cols != cols:
                # Create a new array with the desired dimensions and fill it with zeros
                # adjusted_arr = np.zeros((new_rows, new_cols), dtype=arr.dtype)
                adjusted_arr = np.zeros((new_rows, new_cols), dtype=arr.dtype)
                # Copy the original array into the center of the new array
                adjusted_arr[rows_to_add//2:rows_to_add//2+rows, cols_to_add//2:cols_to_add//2+cols] = arr

                # # Copy the original array into the center of the new array
                # adjusted_arr[:rows, :cols] = arr
                return adjusted_arr
            else:
                # No adjustment needed, return the original array
                return arr
            
def adjust_bit_depth(frame, bit_depth=12):
    """
    Adjusts the bit depth of a raw camera frame.
    
    Parameters:
      frame     : 2D NumPy array containing the raw camera image.
      bit_depth : Bit depth of the raw data (default is 12). If set to 0, defaults to 14.
    
    Returns:
      An 8-bit (grayscale) image with pixel values scaled to the range 0-255.
    """
    # Default to 14-bit if bit_depth is 0
    if bit_depth == 0:
        bit_depth = 14
    levels = 2 ** bit_depth

    # Clamp any negative values to 0
    frame = np.maximum(frame, 0)
    
    # Scale the pixel values to the 8-bit range [0, 255]
    frame_8bit = (frame.astype(np.float32) * 255.0 / (levels - 1)).clip(0, 255).astype(np.uint8)
    
    return frame_8bit



def start_FrameCaptureThread(CamObject:GeneralCameraObject,FrameCaptureThreadFunc):
    if(CamObject.CameraType=="AlliedCamera"):
        
        CamProcess = multiprocessing.Process(target=FrameCaptureThreadFunc, args=(
            CamObject.frame_queue,
            CamObject.Cam_Calibtation,
            CamObject.SetCamCalibrationEvent,
            CamObject.GetFrameFlag,
            CamObject.GetFrameFlag_digholo,
            CamObject.terminateCamera,
            CamObject.FrameObtained,
            CamObject.shm.name,
            CamObject.shm_digholo.name,
            CamObject.FrameHeight,
            CamObject.FrameWidth,
            CamObject.SetGainFlag,CamObject.SetExposureFlag,
            CamObject.ContinuesMode,CamObject.SingleFrameMode,
            CamObject.shared_float,CamObject.shared_int,CamObject.shared_flag_int))
        CamProcess.start()
    elif(CamObject.CameraType=="QImagCamera"):
        CamProcess = multiprocessing.Process(target=FrameCaptureThreadFunc, args=(
            CamObject.frame_queue,
            CamObject.Cam_Calibtation,
            CamObject.SetCamCalibrationEvent,
            CamObject.GetFrameFlag,
            CamObject.GetFrameFlag_digholo,
            CamObject.terminateCamera,
            CamObject.FrameObtained,
            CamObject.shm.name,
            CamObject.shm_digholo.name,
            CamObject.FrameHeight,
            CamObject.FrameWidth,
            CamObject.SetGainFlag,CamObject.SetOffsetFlag,CamObject.SetExposureFlag,
            CamObject.LogPlot,
            CamObject.ContinuesMode,CamObject.SingleFrameMode,
            CamObject.shared_float,CamObject.shared_int,CamObject.shared_flag_int))
        CamProcess.start()
    elif(CamObject.CameraType=="XenicsCamera"):
        CamProcess = multiprocessing.Process(target=FrameCaptureThreadFunc, args=(
            CamObject.frame_queue,
            CamObject.Cam_Calibtation,
            CamObject.SetCamCalibrationEvent,
            CamObject.CameraName,
            CamObject.GetFrameFlag,
            CamObject.GetFrameFlag_digholo,
            CamObject.terminateCamera,
            CamObject.FrameObtained,
            CamObject.shm.name,
            CamObject.shm_digholo.name,
            CamObject.FrameHeight,
            CamObject.FrameWidth,
            CamObject.SetExposureFlag,
            CamObject.SetGainFlag,
            CamObject.SetFanFlag,
            CamObject.ContinuesMode,CamObject.SingleFrameMode,
            CamObject.shared_float,CamObject.shared_int,CamObject.shared_flag_int))
        CamProcess.start()
    elif(CamObject.CameraType=="FLIRCamera"):
        CamProcess = multiprocessing.Process(target=FrameCaptureThreadFunc, args=(
            CamObject.frame_queue,
            CamObject.Cam_Calibtation,
            CamObject.SetCamCalibrationEvent,
            CamObject.GetFrameFlag,
            CamObject.GetFrameFlag_digholo,
            CamObject.terminateCamera,
            CamObject.FrameObtained,
            CamObject.shm.name,
            CamObject.shm_digholo.name,
            CamObject.FrameHeight,
            CamObject.FrameWidth,
            CamObject.SetGainFlag,CamObject.SetExposureFlag,
            CamObject.ContinuesMode,CamObject.SingleFrameMode,
            CamObject.shared_float,CamObject.shared_int,CamObject.shared_flag_int))
        CamProcess.start()
    elif(CamObject.CameraType=="NiTCamera"):
        CamProcess = multiprocessing.Process(target=FrameCaptureThreadFunc, args=(
            CamObject.frame_queue,
            CamObject.Cam_Calibtation,
            CamObject.SetCamCalibrationEvent,
            CamObject.GetFrameFlag,
            CamObject.GetFrameFlag_digholo,
            CamObject.terminateCamera,
            CamObject.FrameObtained,
            CamObject.shm.name,
            CamObject.shm_digholo.name,
            CamObject.FrameHeight,
            CamObject.FrameWidth,
            CamObject.SetGainFlag,CamObject.SetExposureFlag,CamObject.SetCaptureModeFlag,
            CamObject.LogPlot,
            CamObject.ContinuesMode,CamObject.SingleFrameMode,
            CamObject.shared_float,CamObject.shared_int,CamObject.shared_flag_int))
        CamProcess.start()
    else:
        print("Camera type not supported. Write your own code you peasant!!!!")

    return CamProcess

    # def setup_pixel_format(self,cam: Camera):
    #     # Query available pixel formats. Prefer color formats over monochrome formats
    #     cam_formats = cam.get_pixel_formats()
    #     cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
    #     convertible_color_formats = tuple(f for f in cam_color_formats
    #                                     if self.opencv_display_format in f.get_convertible_formats())

    #     cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
    #     convertible_mono_formats = tuple(f for f in cam_mono_formats
    #                                     if self.opencv_display_format in f.get_convertible_formats())

    #     # if OpenCV compatible color format is supported directly, use that
    #     if self.opencv_display_format in cam_formats:
    #         cam.set_pixel_format(self.opencv_display_format)

    #     # else if existing color format can be converted to OpenCV format do that
    #     elif convertible_color_formats:
    #         cam.set_pixel_format(convertible_color_formats[0])

    #     # fall back to a mono format that can be converted
    #     elif convertible_mono_formats:
    #         cam.set_pixel_format(convertible_mono_formats[0])

    #     else:
    #         abort('Camera does not support an OpenCV compatible format. Abort.')

