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

# Function to convert a matplotlib plot to an OpenCV-compatible image
def plot_to_opencv_img(fig):
    """
    Converts a Matplotlib figure to an OpenCV image for display.

    Args:
        fig (matplotlib.figure.Figure): The Matplotlib figure to convert.

    Returns:
        numpy.ndarray: OpenCV-compatible image.
    """
    canvas = FigureCanvas(fig)
    canvas.draw()  # Render the figure onto the canvas
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)  # Get the RGB buffer
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # Reshape to image dimensions
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

def HelloWorldThread(queue, terminateThreadEvent, sharedMemoryNameOscBuffer, sharedMemoryNameTimeArrBuffer, Narr, get_dataEvent, set_AmplitudeEvent, set_ShiftEvent, PlaceHolder):
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
    shmOscBuffer = shared_memory.SharedMemory(name=sharedMemoryNameOscBuffer)
    Osc_buffer = np.ndarray((Narr), dtype=np.float64, buffer=shmOscBuffer.buf)
    
    shmTimeArrBuffer = shared_memory.SharedMemory(name=sharedMemoryNameTimeArrBuffer)
    TimeArr_buffer = np.ndarray((Narr), dtype=np.float64, buffer=shmTimeArrBuffer.buf)

    # Initial message to the queue
    queue.put("Gday_From_The_Queue")

    # Initializing sine wave parameters
    t_arr = np.linspace(0, 2 * np.pi, Narr)
    amplitude = 1
    frequency = 1
    phase = 0
    shift = 0
    Oscdata = amplitude * (np.sin(frequency * t_arr + phase) + shift)
    
    # Set up a Matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    OscDataPlot = ax.plot(t_arr, Oscdata)[0]
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylim([np.min(Oscdata) * 1.1, np.max(Oscdata) * 1.1])
    ax.set_xlim([np.min(t_arr), np.max(t_arr)])
    
    # OpenCV window name
    opencvWindowName = "testData"
    
    # Main loop for updating and displaying the sine wave
    while not terminateThreadEvent.is_set():
        # Update plot data
        OscDataPlot.set_xdata(t_arr)
        OscDataPlot.set_ydata(Oscdata)
        ax.set_xlim(min(t_arr), max(t_arr))
        ax.set_ylim(np.min(Oscdata) - np.abs(np.min(Oscdata)) * 0.2, np.max(Oscdata) + np.abs(np.max(Oscdata) * 0.2))
        fig.canvas.draw_idle()
        imag = plot_to_opencv_img(fig)
        cv2.imshow(opencvWindowName, imag)
        
        # Update sine wave parameters
        phase += np.pi / 10
        Oscdata = amplitude * (np.sin(frequency * t_arr + phase) + shift)
        time.sleep(0.001)  # Small delay to simulate live updating
        
        # Handle shared memory updates
        if get_dataEvent.is_set():
            np.copyto(Osc_buffer, Oscdata)
            np.copyto(TimeArr_buffer, t_arr)
            PlaceHolder.value = phase
            get_dataEvent.clear()
            
        if set_AmplitudeEvent.is_set():
            amplitude = PlaceHolder.value
            set_AmplitudeEvent.clear()
            
        if set_ShiftEvent.is_set():
            shift = PlaceHolder.value
            set_ShiftEvent.clear()
            
        # Check for 'q' key press to terminate
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up OpenCV windows and shared memory
    cv2.destroyAllWindows()
    shmOscBuffer.close()
    shmTimeArrBuffer.close()

    return
class HelloWorldThreadObj:
    """
    Encapsulates a multiprocessing thread for real-time data processing and visualization.

    This class handles the creation of shared memory buffers, synchronization events,
    and the management of a background thread that performs operations such as 
    generating sine wave data and updating parameters dynamically.
    """

    def __init__(self):
        """
        Initializes the thread object, shared memory buffers, and synchronization events.
        """
        # Events for synchronizing actions between processes
        self.get_dataEvent = multiprocessing.Event()  # Signal to retrieve data
        self.set_AmplitudeEvent = multiprocessing.Event()  # Signal to set amplitude
        self.set_ShiftEvent = multiprocessing.Event()  # Signal to set phase shift
        self.terminateThreadEvent = multiprocessing.Event()  # Signal to terminate thread

        # Shared placeholder value for passing parameters between processes
        self.PlaceHolder = multiprocessing.Value('f', 0.0)  # Float value

        # Queue for thread communication
        self.queue = multiprocessing.Queue()

        # Set the size of data arrays (e.g., based on device output, here set to 1024)
        self.Narr = 1024

        # Shared memory buffer for oscilloscope data
        self.sharedMemoryOscBuffer = shared_memory.SharedMemory(create=True, size=int(self.Narr * np.dtype(float).itemsize))
        self.sharedMemoryNameOscBuffer = self.sharedMemoryOscBuffer.name
        self.OscBuffer_shm = np.ndarray((self.Narr), dtype=np.float64, buffer=self.sharedMemoryOscBuffer.buf)

        # Shared memory buffer for time array data
        self.sharedMemoryTimeArrBuffer = shared_memory.SharedMemory(create=True, size=int(self.Narr * np.dtype(float).itemsize))
        self.sharedMemoryNameTimeArrBuffer = self.sharedMemoryTimeArrBuffer.name
        self.TimeArrBuffer_shm = np.ndarray((self.Narr), dtype=np.float64, buffer=self.sharedMemoryTimeArrBuffer.buf)

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

        # Close shared memory buffers
        self.sharedMemoryOscBuffer.close()
        self.sharedMemoryTimeArrBuffer.close()

        # Terminate and join the thread process
        self.Process.terminate()
        self.Process.join()

        # Unlink shared memory buffers to release resources
        self.sharedMemoryOscBuffer.unlink()
        self.sharedMemoryTimeArrBuffer.unlink()

    def start_Thread(self):
        """
        Starts the background thread process.

        The thread runs the `HelloWorldThread` function with required parameters.
        """
        process = multiprocessing.Process(target=HelloWorldThread, args=(
            self.queue,
            self.terminateThreadEvent,
            self.sharedMemoryNameOscBuffer,
            self.sharedMemoryNameTimeArrBuffer,
            self.Narr,
            self.get_dataEvent,
            self.set_AmplitudeEvent,
            self.set_ShiftEvent,
            self.PlaceHolder
        ))
        process.start()  # Start the process
        return process

    def GetData(self):
        """
        Retrieves data from the shared memory buffers.

        This method sets a signal for the thread to update the shared memory,
        waits for the operation to complete, and then returns the data.

        Returns:
            tuple: (time array, oscilloscope data, phase value)
        """
        self.get_dataEvent.set()  # Signal to retrieve data

        # Wait until the thread updates the shared memory
        while self.get_dataEvent.is_set():
            time.sleep(1e-12)

        return self.TimeArrBuffer_shm, self.OscBuffer_shm, self.PlaceHolder.value

    def SetAmplitude(self, Ampvalue):
        """
        Updates the amplitude parameter in the background thread.

        Args:
            Ampvalue (float): The new amplitude value to set.
        """
        self.PlaceHolder.value = Ampvalue  # Update placeholder value
        self.set_AmplitudeEvent.set()  # Signal to update amplitude

        # Wait until the update is completed
        while self.set_AmplitudeEvent.is_set():
            time.sleep(1e-12)
        return

    def SetShift(self, shiftvalue):
        """
        Updates the phase shift parameter in the background thread.

        Args:
            shiftvalue (float): The new shift value to set.
        """
        self.PlaceHolder.value = shiftvalue  # Update placeholder value
        self.set_ShiftEvent.set()  # Signal to update phase shift

        # Wait until the update is completed
        while self.set_ShiftEvent.is_set():
            time.sleep(1e-12)
        return