# import Lab_Equipment.Config.config as config
from Lab_Equipment.Config import config 
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import shared_memory
import copy
import cv2
# import PyCapture2
import time
import ctypes
from QImagCam.qcam import Camera as QImagCamObj
from QImagCam.calibration import get_position, to_wavelength, to_raman

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

# try:
#     import PyCapture2
#     class PointGray_Camera(PyCapture2.Camera): 
#         # Argument in Class Definiation inherit the PyCapture2.Camera class into the Camera Class 
#         # this is equvilent of calling camera = PyCapture2.Camera()
#         def __init__(self):
#             """Initialise camera.

#             Args:
            
#             """
#             super().__init__() # inherit from parent clase PyCapture2.Camera()   
#             # Setup camera
#             # Initialize the camera system
#             bus = PyCapture2.BusManager()
#             # Get the number of connected cameras
#             num_cams = bus.getNumOfCameras()
#             print(f"Number of cameras detected: {num_cams}")
#             if num_cams == 0:
#                 print("No cameras found.")
#                 # return
#             else:
#                 # check if you can connect to camera you may already be connected to camera
#                 # It seems like you can connect to the camera as many times as you want it 
#                 # doesn't matter. I guess you need to becarfull if you want to capture a frame
#                 # and some other process is capturing frames.
#                 try:
#                     self.connect(bus.getCameraFromIndex(0))
#                 except PyCapture2.Fc2error as error:
#                     print("You were already connected to camera: ",error)
                    
#                 # Get a camera frame to set up some properties
#                 Framebuffer=self.Get_Singleframe()
#                 FrameDims=Framebuffer.shape
#                 self.FrameWidth=FrameDims[1]
#                 self.FrameHeight=FrameDims[0]
            
#         def __del__(self):
#             """Destructor to disconnect from camera."""
#             self.disconnect()

#         def StartCaptrue(self):
#             tryCountMax=5
#             tryCount=0
#             success=False
#             while not success:
#                 try:
#                     # try Capture a single frame
#                     image = self.retrieveBuffer()
#                     self.StopCaptrue()
#                     # success = True
#                     print("Camera is already capturing. Capturing was stoped and started again")
#                 except PyCapture2.Fc2error as e:
#                     tryCount=tryCount+1
#                     if (tryCount>tryCountMax):
#                         success = True
#                         self.startCapture()
            
#         def StopCaptrue(self):
#             self.stopCapture()

#         def Get_Singleframe(self):
#             self.StartCaptrue()
            
#             # Loop until we get an image
#             success = False

#             while not success:
#                 try:
#                     # Capture a single frame
#                     FrameRaw = self.retrieveBuffer()
#                     Frame_int = np.array(FrameRaw.getData())
#                     Frame_int = Frame_int.reshape((FrameRaw.getRows(), FrameRaw.getCols(), FrameRaw.getDataSize() // (FrameRaw.getRows() * FrameRaw.getCols())))
#                     Frame=Frame_int.astype(np.float32)
#                     #print("Frame captured and saved.")

#                     success = True
#                 except PyCapture2.Fc2error as e:
#                     print("Error capturing frame in SingleCapture. Trying to get another frame:", e)

#             self.StopCaptrue()

#             return np.squeeze(Frame)

# except ImportError:
#     print("Point gray Camera will not work as you are either using python version above 3.6 or have not installed PyCapture2")

try:
    from vmbpy import *
    # from vimba import * this is the old version 
    class AlliedCameraObject():
        def __init__(self,PixelSize=6.9e-6):
            super().__init__() # inherit from parent class  
            
            # We are just going to go through and grab a bunch of properties from the camera
            # and get 1 frame to set up the shared memory space
            with VmbSystem.get_instance () as vmb:
                cams = vmb.get_all_cameras ()
                with cams [0] as cam:
                    self.print_camera_ID(cam)
                    self.Ny = cam.HeightMax.get()
                    self.Nx = cam.WidthMax.get()
                   
                    print(self.Ny,self.Nx )
                    ExposureTime=cam.ExposureTime.get()
                    # self.ExposureTime=cam.ExposureTime.get()
                    #Note because some compains think it is a good idea to set a camera to not be factor 16 sensor we
                    # we need to check it can adjust if necssary the frame height and width are then adjusted
                    frame = cam.get_frame ()
                    # Frame_int = np.array(frame.as_opencv_image())
                    self.FrameBuffer =adjust_array_dimensions(np.squeeze( np.array(frame.as_opencv_image())))
                    # self.FrameBuffer =adjust_array_dimensions(np.squeeze( np.zeros((self.Ny,self.Nx),dtype=np.uint8)))
                    self.FrameDim=self.FrameBuffer.shape
                    self.FrameHeight = int(self.FrameDim[0])
                    self.FrameWidth = int(self.FrameDim[1])
                    self.Framedtype=self.FrameBuffer.dtype
                    self.PixelSize=PixelSize
  
            # NOTE we need to make a shared memory space for the camera frames for us to access them.
            # This seems to be the most memory efficenct way to do it.   
            # self.shm = shared_memory.SharedMemory(create=True, size=int(np.prod((self.FrameHeight, self.FrameWidth)) * np.dtype(np.uint8).itemsize))
            self.shm = shared_memory.SharedMemory(create=True, size=int(self.FrameHeight* self.FrameWidth * np.dtype(np.uint8).itemsize))
            self.FrameBuffer_SharedMem  = np.ndarray((self.FrameHeight, self.FrameWidth), dtype=self.FrameBuffer.dtype, buffer=self.shm.buf)                    
       
            #We need to set up a bunch of varibles for the threading of the camera.   
            self.terminateCamera = multiprocessing.Event()
            self.GetFrameFlag = multiprocessing.Event()
            self.SetExposureFlag = multiprocessing.Event()
            self.ContinuesMode= multiprocessing.Event()
            self.SingleFrameMode= multiprocessing.Event()



            self.ExposureSet=multiprocessing.Value('i', 0)
            self.ExposureTime=multiprocessing.Value('f', ExposureTime)

            self.FrameObtained = multiprocessing.Value('i', 0)  # Flag to indicate when a frame is ready  
            self.FrameHeightThread=multiprocessing.Value('i', self.FrameHeight)
            self.FrameWidthThread=multiprocessing.Value('i', self.FrameWidth)
            self.frame_queue = multiprocessing.Queue()
    
            # START the camera Thead
            self.CamProcess, self.frame_queue = self.start_FrameCaptureThread()
            
            # self.opencv_display_format = PixelFormat.Bgr8
            
        def __del__(self):
            """Destructor to disconnect from camera."""
            print("Allied Camera Class has been destroyed")
            self.terminateCamera.set()# stop the camera thread
            self.shm.close() # close access to shared memory
            self.CamProcess.terminate()
            self.shm.unlink() # clean up the shared memory space
            
        def setup_pixel_format(self,cam: Camera):
            # Query available pixel formats. Prefer color formats over monochrome formats
            cam_formats = cam.get_pixel_formats()
            cam_color_formats = intersect_pixel_formats(cam_formats, COLOR_PIXEL_FORMATS)
            convertible_color_formats = tuple(f for f in cam_color_formats
                                            if self.opencv_display_format in f.get_convertible_formats())

            cam_mono_formats = intersect_pixel_formats(cam_formats, MONO_PIXEL_FORMATS)
            convertible_mono_formats = tuple(f for f in cam_mono_formats
                                            if self.opencv_display_format in f.get_convertible_formats())

            # if OpenCV compatible color format is supported directly, use that
            if self.opencv_display_format in cam_formats:
                cam.set_pixel_format(self.opencv_display_format)

            # else if existing color format can be converted to OpenCV format do that
            elif convertible_color_formats:
                cam.set_pixel_format(convertible_color_formats[0])

            # fall back to a mono format that can be converted
            elif convertible_mono_formats:
                cam.set_pixel_format(convertible_mono_formats[0])

            else:
                abort('Camera does not support an OpenCV compatible format. Abort.')


        def print_camera_ID(self,cam: Camera):
            print('/// Camera Name   : {}'.format(cam.get_name()))
            print('/// Model Name    : {}'.format(cam.get_model()))
            print('/// Camera ID     : {}'.format(cam.get_id()))
            print('/// Serial Number : {}'.format(cam.get_serial()))
            print('/// Interface ID  : {}\n'.format(cam.get_interface_id()))
        
        def start_FrameCaptureThread(self):
            self.CamProcess = multiprocessing.Process(target=AlliedFrameCaptureThread, args=(
                self.frame_queue,
                self.GetFrameFlag,
                self.terminateCamera,
                self.FrameObtained,
                self.shm.name,
                self.FrameHeight,
                self.FrameWidth,
                self.SetExposureFlag,
                self.ExposureTime,
                self.ExposureSet,
                self.ContinuesMode,self.SingleFrameMode))
            self.CamProcess.start()
            return self.CamProcess, self.frame_queue
    

        def GetFrame(self):            
            #try and grab a frame until you get one.
            while self.FrameObtained.value == 0:
                self.GetFrameFlag.set()
            #Pull the frame from the shared buffer and save it to the camera object buffer so an other process can get to it.
            self.Framebuffer = np.array(self.FrameBuffer_SharedMem)  # Make a copy of the frame
            self.FrameObtained.value = 0
            #NOTE the code below can be used if you wanted to go the route of using the queue instead of sharded memory
            # self.GetFrameFlag.set()
            # while self.frame_queue.empty() == True:
            #     self.GetFrameFlag.set()
            # frame_bytes = self.frame_queue.get_nowait()
            # # self.frame_queue.empty()
            # self.Framebuffer = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(self.FrameDim)
            # self.FrameObtained.value = 0
        def SetExposure(self,NewExposureTime): 
            self.ExposureTime.value=NewExposureTime
            while self.ExposureSet.value == 0:
                self.SetExposureFlag.set()
            self.ExposureSet.value=0

        def SetSingleFrameCapMode(self):
            self.ContinuesMode.clear()
            self.SingleFrameMode.set()
        def SetContinousFrameCapMode(self):
            self.SingleFrameMode.clear()
            self.ContinuesMode.set()

        
    def AlliedFrameCaptureThread(queue,GetFrameFlag,terminateCamFlag,FrameObtained,shared_memory_name,FrameHeight,FrameWidth,SetExposureFlag,ExposureTime,ExposureSet,ContinuesMode,SingleFrameMode):
        # Setup Shared memory
        shm = shared_memory.SharedMemory(name=shared_memory_name)
        frame_buffer = np.ndarray((FrameHeight, FrameWidth), dtype=np.uint8, buffer=shm.buf) 
        ContinuesMode.set()

        with VmbSystem.get_instance () as vmb:
            cams = vmb.get_all_cameras ()
            with cams [0] as cam:
                while not terminateCamFlag.is_set():
                    if (ContinuesMode.is_set()):
                        frame = cam.get_frame ()
                        Frame_int =adjust_array_dimensions(np.squeeze( np.array(frame.as_opencv_image())))                    
                        cv2.imshow("Camera Image", Frame_int)
                        if ( GetFrameFlag.is_set() ):
                            
                            # if (CamInitialised == False):
                            #    CamInitialised=True
                            np.copyto(frame_buffer, Frame_int)
                            FrameObtained.value=1
                            GetFrameFlag.clear()
                            # this was the queue way but it isn't consistant interms of when a frame is obatined
                            # so I have moved to shared memory space method.
                            # frame_bytes = Frame_int.tobytes()
                            # queue.put(frame_bytes)
                    elif(SingleFrameMode.is_set()):
                        if ( GetFrameFlag.is_set() ):
                            frame = cam.get_frame ()
                            Frame_int =adjust_array_dimensions(np.squeeze( np.array(frame.as_opencv_image())))                    
                            cv2.imshow("Camera Image", Frame_int)
                            np.copyto(frame_buffer, Frame_int)
                            FrameObtained.value=1
                            GetFrameFlag.clear()

                    if(SetExposureFlag.is_set()):
                        cam.ExposureTime.set(ExposureTime.value)
                        ExposureSet.value=1
                        SetExposureFlag.clear()


                        

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                cv2.destroyAllWindows()
        shm.close() 
        
    class QImagCamraObject():
        def __init__(self,PixelSize=6.45e-6):
                super().__init__() # inherit from parent class  
                
                # We are just going to go through and grab a bunch of properties from the camera
                # and get 1 frame to set up the shared memory space
                cam = QImagCamObj()
                cameraOpened = cam.connect_to_camera()
                result, model = cam.get_camera_model()
                if result != 0:  # Replace with your actual success value
                    print(f"Failed to get camera model with error {result}")
                else:
                    print(f"Camera model: {model}")

                # Display Info & Parameters
                cam.setup_camera()
                print(list(cam.info.values()))
                print(list(cam.parameters.values()))
                for name, param in cam.parameters.items():
                    print(f"For parameter {name}: min_value is {param.min_value}, max_value is {param.max_value}")
                
                
                self.Ny = cam.QCam_GetInfo(cam.INFO_KEYS["Image Height"])[1]
                self.Nx =cam.QCam_GetInfo(cam.INFO_KEYS["Image Width"])[1]
                ExposureTime=cam.QCam_GetParam(cam.PARAM_KEYS["Exposure"])[1]
                GainTime=cam.QCam_GetParam(cam.PARAM_KEYS["Gain"])[1]
                OffsetTime=cam.QCam_GetParam(cam.PARAM_KEYS["Offset"])[1]

                print('Offset ',OffsetTime)
                print('Gain ',GainTime)
                print('ExposureTime ',ExposureTime)
                print(self.Ny,self.Nx )    

                #Get a frame and see what the size of the frame is
                Rawframe = cam.grab_frame()
                pBuffer = ctypes.cast(Rawframe.pBuffer, ctypes.POINTER(ctypes.c_char * Rawframe.size))
                # Then we create a numpy array from the buffer
                RawFame_data = np.frombuffer(pBuffer.contents, dtype=np.uint8)
                # Now reshape it into the correct shape
                frame = RawFame_data.reshape(Rawframe.height, Rawframe.width) 

                self.FrameBuffer =adjust_array_dimensions(np.squeeze( frame))
                self.FrameDim=self.FrameBuffer.shape
                self.FrameHeight = int(self.FrameDim[0])
                self.FrameWidth = int(self.FrameDim[1])
                self.Framedtype=self.FrameBuffer.dtype
                self.PixelSize=PixelSize
                cam.close_camera()
                cam.release_driver()

    
                # NOTE we need to make a shared memory space for the camera frames for us to access them.
                # This seems to be the most memory efficenct way to do it.   
                # self.shm = shared_memory.SharedMemory(create=True, size=int(np.prod((self.FrameHeight, self.FrameWidth)) * np.dtype(np.uint8).itemsize))
                self.shm = shared_memory.SharedMemory(create=True, size=int(self.FrameHeight* self.FrameWidth * np.dtype(np.uint8).itemsize))
                self.FrameBuffer_SharedMem  = np.ndarray((self.FrameHeight, self.FrameWidth), dtype=self.FrameBuffer.dtype, buffer=self.shm.buf)                    
        
                #We need to set up a bunch of varibles for the threading of the camera.   
                self.terminateCamera = multiprocessing.Event()
                self.GetFrameFlag = multiprocessing.Event()
                self.SetExposureFlag = multiprocessing.Event()
                self.SetGainFlag = multiprocessing.Event()
                self.SetOffsetFlag = multiprocessing.Event()


                self.ContinuesMode= multiprocessing.Event()
                self.SingleFrameMode= multiprocessing.Event()



                self.ExposureSet=multiprocessing.Value('i', 0)
                self.ExposureTime=multiprocessing.Value('f', ExposureTime)
                self.GainSet=multiprocessing.Value('i', 0)
                self.GainTime=multiprocessing.Value('f', GainTime)
                self.OffsetSet=multiprocessing.Value('i', 0)
                self.OffsetTime=multiprocessing.Value('f', OffsetTime)

                self.FrameObtained = multiprocessing.Value('i', 0)  # Flag to indicate when a frame is ready  
                self.FrameHeightThread=multiprocessing.Value('i', self.FrameHeight)
                self.FrameWidthThread=multiprocessing.Value('i', self.FrameWidth)
                self.frame_queue = multiprocessing.Queue()
        
                # START the camera Thead
                self.CamProcess, self.frame_queue = self.start_FrameCaptureThread()
                
                # self.opencv_display_format = PixelFormat.Bgr8
            
        def __del__(self):
            """Destructor to disconnect from camera."""
            print("Allied Camera Class has been destroyed")
            self.terminateCamera.set()# stop the camera thread
            self.shm.close() # close access to shared memory
            self.CamProcess.terminate()
            self.shm.unlink() # clean up the shared memory space
            # cam.close_camera()
            # cam.release_driver()
        def start_FrameCaptureThread(self):
            self.CamProcess = multiprocessing.Process(target=QImagCamFrameCaptureThread, args=(
                self.frame_queue,
                self.GetFrameFlag,
                self.terminateCamera,
                self.FrameObtained,
                self.shm.name,
                self.FrameHeight,
                self.FrameWidth,
                self.SetGainFlag,
                self.GainTime,
                self.GainSet,
                self.SetOffsetFlag,self.OffsetTime,self.OffsetSet,
                self.SetExposureFlag,
                self.ExposureTime,
                self.ExposureSet,
                self.ContinuesMode,self.SingleFrameMode))
            self.CamProcess.start()
            return self.CamProcess, self.frame_queue


        def GetFrame(self):            
            #try and grab a frame until you get one.
            self.GetFrameFlag.set()
            while self.FrameObtained.value == 0:
                time.sleep(1e-9)
                # self.GetFrameFlag.set()
                # print('test')
            #Pull the frame from the shared buffer and save it to the camera object buffer so an other process can get to it.
            self.FrameBuffer= np.array(self.FrameBuffer_SharedMem)  # Make a copy of the frame
            self.FrameObtained.value = 0
            return  self.FrameBuffer
            #NOTE the code below can be used if you wanted to go the route of using the queue instead of sharded memory
            # self.GetFrameFlag.set()
            # while self.frame_queue.empty() == True:
            #     self.GetFrameFlag.set()
            # frame_bytes = self.frame_queue.get_nowait()
            # # self.frame_queue.empty()
            # self.Framebuffer = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(self.FrameDim)
            # self.FrameObtained.value = 0
        def SetExposure(self,NewExposureTime): 
            self.ExposureTime.value=(NewExposureTime)
            while self.ExposureSet.value == 0:
                self.SetExposureFlag.set()

            self.ExposureSet.value=0
            # time.sleep(self.ExposureTime.value*1e-6)
            print('New Exposure time= ',self.ExposureTime.value*1e-6)

        def SetGain(self,NewGainTime): 
            self.GainTime.value=(NewGainTime)
            while self.GainSet.value == 0:
                self.SetGainFlag.set()

            self.GainSet.value=0
            # time.sleep(self.GainTime.value*1e-6)
            print('New Gain time= ',self.GainTime.value)

        def SetOffset(self,NewOffsetTime): 
            self.OffsetTime.value=(NewOffsetTime)
            while self.OffsetSet.value == 0:
                self.SetOffsetFlag.set()

            self.OffsetSet.value=0
            # time.sleep(self.OffsetTime.value*1e-6)
            print('New Offset time= ',self.OffsetTime.value)

        def SetSingleFrameCapMode(self):
            self.ContinuesMode.clear()
            self.SingleFrameMode.set()
        def SetContinousFrameCapMode(self):
            self.SetExposure(100)
            self.SingleFrameMode.clear()
            self.ContinuesMode.set()
            

        
    def QImagCamFrameCaptureThread(queue,GetFrameFlag,terminateCamFlag,FrameObtained,shared_memory_name,FrameHeight,FrameWidth,SetGainFlag,GainTime,GainSet,SetOffsetFlag,OffsetTime,OffsetSet,
                                   SetExposureFlag,ExposureTime,ExposureSet,ContinuesMode,SingleFrameMode):
        # Setup Shared memory
        shm = shared_memory.SharedMemory(name=shared_memory_name)
        frame_buffer = np.ndarray((FrameHeight, FrameWidth), dtype=np.uint8, buffer=shm.buf) 
        ContinuesMode.set()
        # SingleFrameMode.set()
        cam = QImagCamObj()
        cameraOpenedErrorCode = cam.connect_to_camera()
        # queue.put(cameraOpenedErrorCode)

        cam.setup_camera()
        while not terminateCamFlag.is_set():
            
            if (ContinuesMode.is_set()):
                Rawframe=cam.grab_frame()
                """
                Convert frame buffer to image array
                Windows doesn't like it (i.e. throws a fatal exception)
                when this takes place in qcam.py for some reason!
                """
                # First we need to cast the void pointer to a pointer to a char array
                pBuffer = ctypes.cast(Rawframe.pBuffer, ctypes.POINTER(ctypes.c_char * Rawframe.size))
                # Then we create a numpy array from the buffer
                RawFame_data = np.frombuffer(pBuffer.contents, dtype=np.uint8)
                # Now reshape it into the correct shape
                frame = RawFame_data.reshape(Rawframe.height, Rawframe.width) 
                Frame_int =adjust_array_dimensions(np.squeeze( frame))                    
                                 
                cv2.imshow("Camera Image", Frame_int)

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
                    Rawframe = cam.grab_frameSingle()    
                    """
                    Convert frame buffer to image array
                    Windows doesn't like it (i.e. throws a fatal exception)
                    when this takes place in qcam.py for some reason!
                    """
                    # First we need to cast the void pointer to a pointer to a char array
                    pBuffer = ctypes.cast(Rawframe.pBuffer, ctypes.POINTER(ctypes.c_char * Rawframe.size))
                    # Then we create a numpy array from the buffer
                    RawFame_data = np.frombuffer(pBuffer.contents, dtype=np.uint8)
                    # Now reshape it into the correct shape
                    frame = RawFame_data.reshape(Rawframe.height, Rawframe.width) 
                    Frame_int =adjust_array_dimensions(np.squeeze( frame))                    
                    cv2.imshow("Camera Image", Frame_int)
                    np.copyto(frame_buffer, Frame_int)
                    FrameObtained.value=1
                    GetFrameFlag.clear()

            if(SetExposureFlag.is_set()):
                if ( (ExposureTime.value)>=10 and (ExposureTime.value)<=1073741823 ):
                    cam.set_camera_param("Exposure", int(ExposureTime.value))
                    
                ExposureTime.value=int(cam.QCam_GetParam(cam.PARAM_KEYS["Exposure"])[1])
                ExposureSet.value=1
                SetExposureFlag.clear()
            
            if(SetGainFlag.is_set()):
                if ( (GainTime.value)>=115 and (GainTime.value)<=4095):
                    cam.set_camera_param("Gain", int(GainTime.value))
                    
                GainTime.value=int(cam.QCam_GetParam(cam.PARAM_KEYS["Gain"])[1])
                GainSet.value=1
                SetGainFlag.clear()

            if(SetOffsetFlag.is_set()):
                if ( (OffsetTime.value)>=0 and (OffsetTime.value)<=4095):
                    cam.set_camera_param("Offset", int(OffsetTime.value))
                    
                OffsetTime.value=int(cam.QCam_GetParam(cam.PARAM_KEYS["Offset"])[1])
                OffsetSet.value=1
                SetOffsetFlag.clear()

                
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        shm.close()
        cam.close_camera()
        cam.release_driver()
            
         
                
        
        

except ImportError:
    print("Point gray Camera will not work as you are either using python version is not >=3.7 or have not installed vmbpy")
    
# def camera_process(shared_memory_name, frame_ready, lock):
#     shm = shared_memory.SharedMemory(name=shared_memory_name)
#     frame_shape = (480, 640, 3)  # Example frame shape
#     frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)

#     cap = cv2.VideoCapture(0)
#     while True:
#         ret, frame = cap.read()
#         if ret:
#             with lock:  # Acquire the lock before accessing shared memory
#                 np.copyto(frame_buffer, frame)
#             frame_ready.value = 1

#         time.sleep(0.01)  # Sleep to prevent a tight loop

#     cap.release()
#     shm.close()

# def main_process(shared_memory_name, frame_ready, lock):
#     shm = shared_memory.SharedMemory(name=shared_memory_name)
#     frame_shape = (480, 640, 3)  # Example frame shape
#     frame_buffer = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)

#     while True:
#         if frame_ready.value == 1:
#             with lock:  # Acquire the lock before accessing shared memory
#                 frame = np.array(frame_buffer)  # Make a copy of the frame
#             # Process the frame...
#             frame_ready.value = 0

#         time.sleep(0.01)  # Sleep to prevent a tight loop

#     shm.close()

# if __name__ == "__main__":
#     frame_shape = (480, 640, 3)
#     shm = shared_memory.SharedMemory(create=True, size=np.prod(frame_shape) * np.dtype(np.uint8).itemsize)
#     frame_ready = multiprocessing.Value('i', 0)  # Flag to indicate when a frame is ready
#     lock = multiprocessing.Lock()  # Lock for synchronizing access to the shared memory

#     p1 = multiprocessing.Process(target=camera_process, args=(shm.name, frame_ready, lock))
#     p2 = multiprocessing.Process(target=main_process, args=(shm.name, frame_ready, lock))

#     p1.start()
#     p2.start()

#     p1.join()
#     p2.join()

#     shm.close()
#     shm.unlink()  # Free the shared memory