# Importing necessary libraries
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import multiprocessing
from multiprocessing import shared_memory
import time
import copy
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from screeninfo import get_monitors


class FullScreenDisplayObject:
    """
    Encapsulates a multiprocessing thread for real-time data processing and visualization.

    This class handles the creation of shared memory buffers, synchronization events,
    and the management of a background thread that performs operations such as 
    generating sine wave data and updating parameters dynamically.
    """

    def __init__(self,monitor_index=1,RefreshRate=500e-3):
        """
        Initializes the thread object, shared memory buffers, and synchronization events.
        """
        # Events for synchronizing actions between processes
        self.UpdateDisplay = multiprocessing.Event()  # Signal to retrieve data
        self.terminateThreadEvent = multiprocessing.Event()  # Signal to terminate thread

        # Queue for thread communication
        self.queue = multiprocessing.Queue()

        self.monitor_x,self.monitor_y,self.monitor_height, self.monitor_width=opencv_display_on_monitor(monitor_index)

        # Shared memory buffer for oscilloscope data
        self.sharedMemoryDisplayBuffer = shared_memory.SharedMemory(create=True, size=int(self.monitor_height* self.monitor_width* 3 * np.dtype(np.uint8).itemsize))
        self.sharedMemoryDisplayBufferName = self.sharedMemoryDisplayBuffer.name
        self.DisplayBuffer_arr_shm = np.ndarray((self.monitor_height, self.monitor_width, 3), dtype=np.uint8, buffer=self.sharedMemoryDisplayBuffer.buf)
        self.RefreshRate=multiprocessing.Value("f",RefreshRate)
      
        # Start the thread process
        self.Process = self.start_Thread()

    def __del__(self):
        """
        Destructor to clean up resources when the object is deleted.

        This ensures that the background thread is terminated, shared memory buffers are closed,
        and all allocated resources are released.
        """
        print("Cleaning up resources...")
        self.terminateThreadEvent.set()  # Signal thread termination
        time.sleep(1)
        # Close shared memory buffers
        self.sharedMemoryDisplayBuffer.close()

        # Terminate and join the thread process
        self.Process.terminate()
        self.Process.join()

        # Unlink shared memory buffers to release resources
        self.sharedMemoryDisplayBuffer.unlink()
        print("Destroying")

    def start_Thread(self):
        """
        Starts the background thread process.

        The thread runs the `HelloWorldThread` function with required parameters.
        """
        process = multiprocessing.Process(target=SLMScreenDisplayThread, args=(
            self.queue,
            self.terminateThreadEvent,
            self.RefreshRate,
            self.UpdateDisplay,
            self.sharedMemoryDisplayBufferName,
             self.monitor_x,
             self.monitor_y,
             self.monitor_height, 
             self.monitor_width
        ))
        process.start()  # Start the process
        return process

    def Send_Image_To_Display(self,channelIdx,NewImage=None):
        if NewImage is not None:
            NewImage.shape
            if  (NewImage.shape[0] == self.monitor_height and NewImage.shape[1]== self.monitor_width):
                
                np.copyto(self.DisplayBuffer_arr_shm[:,:,channelIdx],NewImage)
                self.UpdateDisplay.set()  # Signal to retrieve data
                # Wait until the thread updates the shared memory
                while self.UpdateDisplay.is_set():
                    time.sleep(1e-12)
                time.sleep(self.RefreshRate.value)
            else:
                print("New image incorrect dimensions for screen display. Display not updated")
        else:
            print("No image sent")
            return 
             
        
            
    def Set_RefreshRate(self,NewRefreshRate):
        self.RefreshRate.value=NewRefreshRate
            
def opencv_display_on_monitor(monitor_index=0):
    # Retrieve information about all connected monitors
    monitors = get_monitors()
    if monitor_index >= len(monitors):
        print(f"Monitor index {monitor_index} out of range. Using primary monitor instead.")
        monitor = monitors[0]
    else:
        monitor = monitors[monitor_index]

    # Get the monitor's position and dimensions
    monitor_x = monitor.x
    monitor_y = monitor.y
    monitor_width = monitor.width
    monitor_height = monitor.height
    print(f"Using monitor {monitor_index}: x={monitor_x}, y={monitor_y}, width={monitor_width}, height={monitor_height}")
    return monitor_x,monitor_y,monitor_height, monitor_width

def SLMScreenDisplayThread(queue, terminateThreadEvent,RefreshRate,UpdateDisplay, sharedMemoryNameDisplayBuffer,
                           monitor_x,monitor_y,monitor_height, monitor_width):
    """
    A multiprocessing thread function to generate and display a sine wave.

    Args:
        queue (multiprocessing.Queue): Queue for thread communication.
        terminateThreadEvent (multiprocessing.Event): Event to signal thread termination.
        sharedMemoryNameOscBuffer (str): Shared memory name for oscilloscope buffer.
        sharedMemoryNameTimeArrBuffer (str): Shared memory name for time array buffer.
        Narr (int): Number of data points in arrays.
        get_dataEvent (multiprocessing.Event): Event to signal data retrieval.
        set_AmplitudeEvent (multiprocessing.Event): Event to signal amplitude update.
        set_ShiftEvent (multiprocessing.Event): Event to signal shift update.
        PlaceHolder (multiprocessing.Value): Shared placeholder for parameter updates.
    """
    # Access shared memory buffers
    DisplayBuffer = shared_memory.SharedMemory(name=sharedMemoryNameDisplayBuffer)
    DisplayBuffer_arr_shm = np.ndarray((monitor_height, monitor_width, 3), dtype=np.uint8, buffer=DisplayBuffer.buf)
    opencvWindowName = "SLMFullScreen"
    opencvWindowName_preview = "SLMPreviewScreen"
    scale=1/3
    # Create a window that we can position manually
    cv2.namedWindow(opencvWindowName, cv2.WINDOW_NORMAL)
    cv2.namedWindow(opencvWindowName_preview, cv2.WINDOW_NORMAL)

    # Set window to full screen mode if desired
    cv2.setWindowProperty(opencvWindowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.resizeWindow(opencvWindowName_preview, int(monitor_width*scale), int(monitor_height*scale))
   # cv2.resizeWindow("Display", monitor_width, monitor_height)
    # Move the window to the monitor's position
    cv2.moveWindow(opencvWindowName, monitor_x, monitor_y)
    cv2.imshow(opencvWindowName, DisplayBuffer_arr_shm)
    cv2.imshow(opencvWindowName_preview, DisplayBuffer_arr_shm)

    # Main loop for updating and displaying the sine wave
    while not terminateThreadEvent.is_set():
       
        # Handle shared memory updates
        if UpdateDisplay.is_set():
            # time.sleep(RefreshRate.value)
            cv2.imshow(opencvWindowName, DisplayBuffer_arr_shm)
            cv2.imshow(opencvWindowName_preview, DisplayBuffer_arr_shm)

            # time.sleep(RefreshRate.value)
            UpdateDisplay.clear()
            
        # Break loop when the ESC key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up OpenCV windows and shared memory
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    
    DisplayBuffer.close()

    return