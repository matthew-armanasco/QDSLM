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
import os
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
                cam.ExposureTime.set(28)
                Exposure=cam.ExposureTime.get()
                
                # Not sure whether this camera has gain and offset term
                # Gain=cam.Gain.get()
                # Offset=cam.Offset.get()

                # self.ExposureTime=cam.ExposureTime.get()
                #Note because some companies think it is a good idea to set a camera to not be factor 16 sensor we
                # we need to check it can adjust if necssary the frame height and width are then adjusted
                frame = cam.get_frame ()
                # Frame_int = np.array(frame.as_opencv_image())
                FrameBuffer =CamForm.adjust_array_dimensions(np.squeeze( np.array(frame.as_opencv_image())))
                # self.FrameBuffer =adjust_array_dimensions(np.squeeze( np.zeros((self.Ny,self.Nx),dtype=np.uint8)))
                FrameDim=FrameBuffer.shape
                FrameHeight = int(FrameDim[0])
                FrameWidth = int(FrameDim[1])
                Framedtype=str(FrameBuffer.dtype)
                
        self.CamObject=CamForm.GeneralCameraObject("AlliedCamera",'',self.Nx,self.Ny,
                                                   FrameWidth,FrameHeight,FrameDim,Framedtype,
                                                   FrameBuffer,PixelSize,Exposure,Offset=0,Gain=0)
        self.CamProcess= CamForm.start_FrameCaptureThread(self.CamObject,AlliedFrameCaptureThread)
        # self.start_FrameCaptureThread()
        
        
    def __del__(self):
        """Destructor to disconnect from camera."""
        print(self.CamObject.CameraType +" Class has been destroyed")
        self.CamObject.terminateCamera.set()# stop the camera thread
        self.CamObject.shm.close() # close access to shared memory
        self.CamProcess.terminate()
        self.CamObject.shm.unlink() # clean up the shared memory space
        
    def print_camera_ID(self,cam: Camera):
        print('/// Camera Name   : {}'.format(cam.get_name()))
        print('/// Model Name    : {}'.format(cam.get_model()))
        print('/// Camera ID     : {}'.format(cam.get_id()))
        print('/// Serial Number : {}'.format(cam.get_serial()))
        print('/// Interface ID  : {}\n'.format(cam.get_interface_id()))
        
def AlliedFrameCaptureThread(queue,Cam_Calibtation,SetCalibrationEvent,
                             GetFrameFlag,GetFrameFlag_digholo,terminateCamFlag,FrameObtained,shared_memory_name,shared_memory_name_digholo,FrameHeight,FrameWidth,
                             SetGainFlag,SetExposureFlag,ContinuesMode,SingleFrameMode,
                                   shared_float,shared_int,shared_flag_int):
    # Setup Shared memory
    shm = shared_memory.SharedMemory(name=shared_memory_name)
    frame_buffer = np.ndarray((FrameHeight, FrameWidth), dtype=np.uint8, buffer=shm.buf) 
    shm_digholo = shared_memory.SharedMemory(name=shared_memory_name_digholo)
    frame_buffer_digholo = np.ndarray((FrameHeight, FrameWidth), dtype=np.uint8, buffer=shm_digholo.buf) 
    
    ContinuesMode.set()
    opencvWindowName="Allied Camera Image"
    with VmbSystem.get_instance () as vmb:
        cams = vmb.get_all_cameras ()
        with cams [0] as cam:
            while not terminateCamFlag.is_set():
                if (ContinuesMode.is_set()):
                    frame = cam.get_frame ()
                    Frame_int =CamForm.adjust_array_dimensions(np.squeeze( np.array(frame.as_opencv_image())))                    
                    cv2.imshow(opencvWindowName, Frame_int)
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
                    # I am not really worried about getting the latest frame just want to see something updating on the digholo
                    if ( GetFrameFlag_digholo.is_set() ): 
                        np.copyto(frame_buffer_digholo, Frame_int)
                        GetFrameFlag_digholo.clear()
                elif(SingleFrameMode.is_set()):
                    if ( GetFrameFlag.is_set() ):
                        frame = cam.get_frame ()
                        Frame_int =CamForm.adjust_array_dimensions(np.squeeze( np.array(frame.as_opencv_image())))                    
                        cv2.imshow(opencvWindowName, Frame_int)
                        np.copyto(frame_buffer, Frame_int)
                        FrameObtained.value=1
                        GetFrameFlag.clear()
                    # I am not really worried about getting the latest frame just want to see something updating on the digholo
                    if ( GetFrameFlag_digholo.is_set() ): 
                        np.copyto(frame_buffer_digholo, Frame_int)
                        GetFrameFlag_digholo.clear()

                if(SetExposureFlag.is_set()):
                    Exposure=shared_float.value
                    if ( (Exposure)>=28 and (Exposure)<2e6 ):
                        cam.ExposureTime.set(Exposure)
                        shared_float.value=cam.ExposureTime.get()
                        shared_flag_int.value=1
                        SetExposureFlag.clear()

                if(SetGainFlag.is_set()):
                    Gain=shared_float.value
                    cam.Gain.set(Gain)
                    shared_float.value=cam.Gain.get()
                    shared_flag_int.value=1
                    SetGainFlag.clear()
                if(SetCalibrationEvent.is_set()):
                    CalibrationFile=Cam_Calibtation['CalibrationFile']
                    if os.path.exists(CalibrationFile):
                        shared_int.value=0
                    else:
                        shared_int.value=-1
                    # I am not sure how to set the calibration file on this camera i will work it out when i need to
                    shared_int.value=-1
                    SetCalibrationEvent.clear()
            


                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
    shm.close()
    shm_digholo.close() 
    
 