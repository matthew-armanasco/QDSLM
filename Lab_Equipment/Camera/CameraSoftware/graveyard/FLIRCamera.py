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
import Lab_Equipment.Camera.CameraObject as CamForm
# import os
# # Allow multiple OpenMP runtimes (not recommended for production)
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# NOTE you need to move the libiomp5md.dll out on the bin64 folder where the FlyCapture2_C_v100.dll is located. It confuses python a lot and makes the kernal crash.
fc2lib = ctypes.cdll.LoadLibrary("C:\\Program Files\\Point Grey Research\\FlyCapture2\\bin64\\FlyCapture2_C_v100.dll")

class fc2PixelFormat(ctypes.c_uint):
    FC2_PIXEL_FORMAT_MONO8 = 0x80000000
    FC2_PIXEL_FORMAT_411YUV8 = 0x40000000
    FC2_PIXEL_FORMAT_422YUV8 = 0x20000000
    FC2_PIXEL_FORMAT_444YUV8 = 0x10000000
    FC2_PIXEL_FORMAT_RGB8 = 0x08000000
    FC2_PIXEL_FORMAT_MONO16 = 0x04000000
    FC2_PIXEL_FORMAT_RGB16 = 0x02000000
    FC2_PIXEL_FORMAT_S_MONO16 = 0x01000000
    FC2_PIXEL_FORMAT_S_RGB16 = 0x00800000
    FC2_PIXEL_FORMAT_RAW8 = 0x00400000
    FC2_PIXEL_FORMAT_RAW16 = 0x00200000
    FC2_PIXEL_FORMAT_MONO12 = 0x00100000
    FC2_PIXEL_FORMAT_RAW12 = 0x00080000
    FC2_PIXEL_FORMAT_BGR = 0x80000008
    FC2_PIXEL_FORMAT_BGRU = 0x40000008
    FC2_PIXEL_FORMAT_RGB = 0x08000000  # Same value as FC2_PIXEL_FORMAT_RGB8
    FC2_PIXEL_FORMAT_RGBU = 0x40000002
    FC2_PIXEL_FORMAT_BGR16 = 0x02000001
    FC2_PIXEL_FORMAT_BGRU16 = 0x02000002
    FC2_PIXEL_FORMAT_422YUV8_JPEG = 0x40000001
    FC2_NUM_PIXEL_FORMATS = 20
    FC2_UNSPECIFIED_PIXEL_FORMAT = 0
    
class fc2BayerTileFormat(ctypes.c_uint):
    FC2_BT_NONE = 0
    FC2_BT_RGGB = 1
    FC2_BT_GRBG = 2
    FC2_BT_GBRG = 3
    FC2_BT_BGGR = 4
    # Assuming FULL_32BIT_VALUE is defined elsewhere or you can define it based on your requirements
    # For example, if FULL_32BIT_VALUE is not defined elsewhere, you could explicitly set it as an arbitrary 32-bit value:
    FC2_BT_FORCE_32BITS = 0x80000000  # Example value, adjust as necessary
    
# Define fc2ImageImpl as a generic pointer type
# fc2ImageImpl = ctypes.c_void_p

class fc2Image(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_uint),
        ("cols", ctypes.c_uint),
        ("stride", ctypes.c_uint),
        ("pData", ctypes.POINTER(ctypes.c_ubyte)),
        ("dataSize", ctypes.c_uint),
        ("receivedDataSize", ctypes.c_uint),
        ("format", fc2PixelFormat),
        ("bayerFormat", fc2BayerTileFormat),
        ("imageImpl", ctypes.c_void_p),
    ]
fc2Guid = (ctypes.c_uint * 4)()

class fc2PropertyType(ctypes.c_uint):
    FC2_BRIGHTNESS = 0
    FC2_AUTO_EXPOSURE = 1
    FC2_SHARPNESS = 2
    FC2_WHITE_BALANCE = 3
    FC2_HUE = 4
    FC2_SATURATION = 5
    FC2_GAMMA = 6
    FC2_IRIS = 7
    FC2_FOCUS = 8
    FC2_ZOOM = 9
    FC2_PAN = 10
    FC2_TILT = 11
    FC2_SHUTTER = 12
    FC2_GAIN = 13
    FC2_TRIGGER_MODE = 14
    FC2_TRIGGER_DELAY = 15
    FC2_FRAME_RATE = 16
    FC2_TEMPERATURE = 17
    FC2_UNSPECIFIED_PROPERTY_TYPE = 18
    # Assuming FULL_32BIT_VALUE is defined elsewhere with the correct value
    # Placeholder for demonstration, replace with actual value as necessary
    FC2_PROPERTY_TYPE_FORCE_32BITS = 0x80000000
# NOTE even though some of the varible in fc2Property fc2PropertyInfo are bool ctypes seem to have trouble with them so I have made them int instead   
class fc2Property(ctypes.Structure):
    _fields_ = [
        ("type", fc2PropertyType),
        ("present",  ctypes.c_int),  # Assuming BOOL is equivalent to int
        ("absControl",  ctypes.c_int),
        ("onePush",  ctypes.c_int),
        ("onOff",  ctypes.c_int),
        ("autoManualMode",  ctypes.c_int),
        ("valueA", ctypes.c_uint),
        ("valueB", ctypes.c_uint),
        ("absValue", ctypes.c_float),
        ("reserved", ctypes.c_uint * 8),  # Array of 8 unsigned ints
    ]
    
MAX_STRING_LENGTH =512

class fc2PropertyInfo(ctypes.Structure):
    _fields_ = [
        ("type", fc2PropertyType),  # Assuming fc2PropertyType is an enum/int
        ("present", ctypes.c_int),
        ("autoSupported", ctypes.c_int),
        ("manualSupported", ctypes.c_int),
        ("onOffSupported", ctypes.c_int),
        ("onePushSupported", ctypes.c_int),
        ("absValSupported", ctypes.c_int),
        ("readOutSupported", ctypes.c_int),
        ("min", ctypes.c_uint),
        ("max", ctypes.c_uint),
        ("absMin", ctypes.c_float),
        ("absMax", ctypes.c_float),
        ("pUnits", ctypes.c_char * MAX_STRING_LENGTH),
        ("pUnitAbbr", ctypes.c_char * MAX_STRING_LENGTH),
        ("reserved", ctypes.c_uint * 8),
    ]

# Setup fc2CreateContext function prototype in Python
# fc2lib.fc2CreateContext.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
fc2lib.fc2CreateContext.argtypes = [ctypes.c_void_p]

fc2lib.fc2CreateContext.restype = ctypes.c_int  # Assuming fc2Error is an integer
fc2lib.fc2GetNumOfCameras.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_uint)]
fc2lib.fc2GetNumOfCameras.restype = ctypes.c_int  # Assuming fc2Error is an integer
fc2lib.fc2DestroyContext.argtypes=[ctypes.c_void_p]

# Set up the argument and return types for fc2CreateImage
fc2lib.fc2CreateImage.argtypes = [ctypes.POINTER(fc2Image)]
fc2lib.fc2CreateImage.restype = ctypes.c_int  # Assuming fc2Error is an int
fc2lib.fc2DestroyImage.argtypes = [ctypes.POINTER(fc2Image)]
fc2lib.fc2DestroyImage.restype = ctypes.c_int  

fc2lib.fc2RetrieveBuffer.argtypes = [ctypes.c_void_p,ctypes.POINTER(fc2Image)]
fc2lib.fc2RetrieveBuffer.restype = ctypes.c_int  

fc2lib.fc2GetCameraFromIndex.argtypese= [ctypes.c_void_p,ctypes.c_uint,ctypes.POINTER(fc2Image)];
fc2lib.fc2GetCameraFromIndex.restype = ctypes.c_int  

fc2lib.fc2Connect.argtypese= [ctypes.c_void_p,ctypes.POINTER(fc2Image)];
fc2lib.fc2Connect.restype = ctypes.c_int  

fc2lib.fc2StartCapture.argtypese= [ctypes.c_void_p];
fc2lib.fc2StartCapture.restype = ctypes.c_int  

fc2lib.fc2StopCapture.argtypese= [ctypes.c_void_p];
fc2lib.fc2StopCapture.restype = ctypes.c_int  

fc2lib.fc2GetProperty.argtypese= [ctypes.c_void_p,ctypes.POINTER(fc2Property)];
fc2lib.fc2GetProperty.restype = ctypes.c_int  
fc2lib.fc2SetProperty.argtypese= [ctypes.c_void_p,ctypes.POINTER(fc2Property)];
fc2lib.fc2SetProperty.restype = ctypes.c_int  

error =fc2lib.fc2SaveImage.argtypese= [ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int];
fc2lib.fc2SaveImage.restype = ctypes.c_int  

fc2lib.fc2GetPropertyInfo.argtypese= [ctypes.c_void_p,ctypes.POINTER(fc2PropertyInfo)];
fc2lib.fc2GetPropertyInfo.restype = ctypes.c_int  


class FLIRCameraObject():
    def __init__(self,PixelSize=6.9e-6):
        super().__init__() # inherit from parent class
        # Need to make a context
        context = ctypes.c_void_p(0)
        error_code = fc2lib.fc2CreateContext(ctypes.byref(context))
        #see if there are any Cameras 
        numCameras=ctypes.c_uint()
        error = fc2lib.fc2GetNumOfCameras((context), ctypes.byref(numCameras));
        #Get a ID for the camera 
        index=ctypes.c_uint(0)
        error = fc2lib.fc2GetCameraFromIndex(context, index, ctypes.byref(fc2Guid));
        # connect to the camera 
        error = fc2lib.fc2Connect(context, ctypes.byref(fc2Guid));
        #Start a capature
        error = fc2lib.fc2StartCapture(context);
        
        #get the current properties
        Properties=fc2Property()
        Properties.type=fc2PropertyType.FC2_GAIN
        fc2lib.fc2GetProperty(context,ctypes.byref(Properties) );
        Properties.autoManualMode=0 # Need to switch the Auto off so that it can be changed
        Gain=Properties.absValue
        Properties.type=fc2PropertyType.FC2_SHUTTER #Shutter is the same as the Exposure time on all other cameras
        fc2lib.fc2GetProperty(context,ctypes.byref(Properties) );
        Properties.autoManualMode=0 # Need to switch the Auto off so that it can be changed
        Exposure=Properties.absValue
        
        # Need to make this c structure to which the image and some of the other information will be in
        rawImage = fc2Image() #fc2Image rawImage;
        # Call fc2CreateImage, passing a reference to rawImage this initallises all the c variables in rawImage structure
        error = fc2lib.fc2CreateImage(ctypes.byref(rawImage))
        # Now we can actually get a frame some times you will ask for a fame too quickly so you need to keep asking until you get one
        error = fc2lib.fc2RetrieveBuffer(context, ctypes.byref(rawImage));
        while (error!=0):# just keep trying to grab a frame until you get one
            error = fc2lib.fc2RetrieveBuffer(context, ctypes.byref(rawImage));
            
        # convert the c stuff to python stuff essentially taking the pointer and making it into a np array    
        frameBufferPtr=rawImage.pData
        self.Ny = rawImage.cols
        self.Nx = rawImage.rows
        Framedtype=np.uint8
        frame = np.ctypeslib.as_array(frameBufferPtr,shape=(self.Nx,self.Ny))
        
        FrameDim=frame.shape
        FrameHeight = int(FrameDim[0])
        FrameWidth = int(FrameDim[1])
        print(FrameDim)
        FrameBuffer =CamForm.adjust_array_dimensions(np.squeeze( (frame)))
        
        
        error = fc2lib.fc2DestroyImage(ctypes.byref(rawImage));
        # Need to stop capture and destory the connection to the context
        error = fc2lib.fc2StopCapture(context);
        fc2lib.fc2DestroyContext(context)
        self.CamObject=CamForm.GeneralCameraObject("FLIRCamera",self.Nx,self.Ny,FrameWidth,FrameHeight,FrameDim,Framedtype,FrameBuffer,PixelSize,Exposure,0,Gain,'')
        self.CamProcess= CamForm.start_FrameCaptureThread(self.CamObject,FLIRFrameCaptureThread)
        
        
    def __del__(self):
        """Destructor to disconnect from camera."""
        print(self.CamObject.CameraType +" Class has been destroyed")
        self.CamObject.terminateCamera.set()# stop the camera thread
        self.CamObject.shm.close() # close access to shared memory
        self.CamProcess.terminate()
        self.CamObject.shm.unlink() # clean up the shared memory space
        
def FLIRFrameCaptureThread(queue,GetFrameFlag,terminateCamFlag,FrameObtained,shared_memory_name,FrameHeight,FrameWidth,SetGainFlag,Gain,GainSet,
                                   SetExposureFlag,Exposure,ExposureSet,ContinuesMode,SingleFrameMode):
    # Setup Shared memory
    shm = shared_memory.SharedMemory(name=shared_memory_name)
    frame_buffer = np.ndarray((FrameHeight, FrameWidth), dtype=np.uint8, buffer=shm.buf) 
    ContinuesMode.set()
    opencvWindowName="FLIR Camera Image"
    Properties=fc2Property() # need this to access the and change the properties of the camera
    context = ctypes.c_void_p(0)
    error_code = fc2lib.fc2CreateContext(ctypes.byref(context))
    #see if there are any Cameras 
    numCameras=ctypes.c_uint()
    error = fc2lib.fc2GetNumOfCameras((context), ctypes.byref(numCameras));
    #Get a ID for the camera 
    index=ctypes.c_uint(0)
    error = fc2lib.fc2GetCameraFromIndex(context, index, ctypes.byref(fc2Guid));
    # connect to the camera 
    error = fc2lib.fc2Connect(context, ctypes.byref(fc2Guid));
    #Start a capature
    error = fc2lib.fc2StartCapture(context);
    # Need to make this c structure to which the image and some of the other information will be in
    rawImage = fc2Image() #fc2Image rawImage;
    # Call fc2CreateImage, passing a reference to rawImage this initallises all the c variables in rawImage structure
    error = fc2lib.fc2CreateImage(ctypes.byref(rawImage))
    
    while not terminateCamFlag.is_set():
        if (ContinuesMode.is_set()):
            error = fc2lib.fc2RetrieveBuffer(context, ctypes.byref(rawImage));
            while (error!=0):# just keep trying to grab a frame until you get one
                error = fc2lib.fc2RetrieveBuffer(context, ctypes.byref(rawImage));
            frameBufferPtr=rawImage.pData
            frame = np.ctypeslib.as_array(frameBufferPtr,shape=(FrameHeight,FrameWidth))
            Frame_int =CamForm.adjust_array_dimensions(frame)
            cv2.imshow(opencvWindowName, Frame_int)
            
            if ( GetFrameFlag.is_set() ):
                np.copyto(frame_buffer, Frame_int)
                FrameObtained.value=1
                GetFrameFlag.clear()
                # this was the queue way but it isn't consistant interms of when a frame is obatined
                # so I have moved to shared memory space method.
                # frame_bytes = Frame_int.tobytes()
                # queue.put(frame_bytes)
        elif(SingleFrameMode.is_set()):
            if ( GetFrameFlag.is_set() ):
                
                error = fc2lib.fc2RetrieveBuffer(context, ctypes.byref(rawImage));
                while (error!=0):# just keep trying to grab a frame until you get one
                    error = fc2lib.fc2RetrieveBuffer(context, ctypes.byref(rawImage));
                frameBufferPtr=rawImage.pData
                frame = np.ctypeslib.as_array(frameBufferPtr,shape=(FrameHeight,FrameWidth))
                Frame_int =CamForm.adjust_array_dimensions(frame)
                cv2.imshow(opencvWindowName, Frame_int)
                np.copyto(frame_buffer, Frame_int)
                FrameObtained.value=1
                GetFrameFlag.clear()

        if(SetExposureFlag.is_set()):
            if ( (Exposure.value)>=0.005125999450683594 and (Exposure.value)<61.5157470703125):#milliseconds
                Properties.type=fc2PropertyType.FC2_SHUTTER #Shutter is the same as the Exposure time on all other cameras
                # fc2lib.fc2GetProperty(context,ctypes.byref(Properties) );
                Properties.autoManualMode=0 # Need to switch the Auto off so that it can be changed
                Properties.absValue=Exposure.value
                error=fc2lib.fc2SetProperty(context,ctypes.byref(Properties) );
                ExposureSet.value=1
                fc2lib.fc2GetProperty(context,ctypes.byref(Properties) );
                Exposure.value=Properties.absValue
                SetExposureFlag.clear()

        if(SetGainFlag.is_set()):
            if ( (Gain.value)>=-5.630345344543457 and (Gain.value)<24.00007438659668): #dB
                Properties.type=fc2PropertyType.FC2_GAIN #Shutter is the same as the Exposure time on all other cameras
                Properties.autoManualMode=0 # Need to switch the Auto off so that it can be changed
                Properties.absValue=Gain.value
                error=fc2lib.fc2SetProperty(context,ctypes.byref(Properties) );
                GainSet.value=1
                fc2lib.fc2GetProperty(context,ctypes.byref(Properties) );
                Gain.value=Properties.absValue
                SetGainFlag.clear()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    shm.close() 
     
        