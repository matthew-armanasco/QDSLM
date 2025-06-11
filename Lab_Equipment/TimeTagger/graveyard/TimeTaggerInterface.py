from Lab_Equipment.Config import config
import numpy as np
import time
import multiprocessing
from multiprocessing import shared_memory
import TimeTagger
import cv2

"""Important notes: 

1. You can't start a multiprocessing process from
within a Jupyter notebook. You have to put the call to start the
process in its own function.

2. The process function cannot be a method in a class. When a process
is created, there is a bunch of memory that has to be duplicated
in a procedure called pickling. If self is an argument to the 
function, then the process manager has to pickle the class
which might not be pickleable.

See https://dannyvanpoucke.be/parallel-python-classes-pickle/.
"""

WAIT_FOR_THREAD_TIME = 1e-12

SUCCESS_CODE = 0
UNKNOWN_ERROR_CODE = 1
TAGGER_NOT_CONNECTED_CODE = 2

EXIT_MESSAGE = {
    SUCCESS_CODE : "exited successfully",
    UNKNOWN_ERROR_CODE : "probably crashed",
    TAGGER_NOT_CONNECTED_CODE : "couldn't connect to tagger"
}

IMAGE_X = 1600
IMAGE_Y = 700

def displayCounts(channel1Counts,channel2Counts,coincidences,coincidencesAverage,
                  coincidencesDeviation, contrast, contrastAverage,
                  contrastDeviation):
    # Create a blank 512x512 black image (3 channels for RGB, uint8)
    image = np.zeros((IMAGE_Y, IMAGE_X, 3), np.uint8)

    # Values to display
    values = {
        'Ch1': str(channel1Counts),
        'Ch2': str(channel2Counts),
        'Coin': '{} / {} ({})'.format(coincidences, coincidencesAverage, coincidencesDeviation),
        'Cont': '{} / {} ({})'.format(contrast, contrastAverage, contrastDeviation)
    }

    # Set the font
    font = cv2.FONT_HERSHEY_DUPLEX

    # Starting position for the first item
    start_x, start_y = 10, 140

    # Display each value on the image
    for i, (key, value) in enumerate(values.items()):
        text = f"{key}: {value}"
        cv2.putText(image, text, (start_x, start_y + i * 140), font, 4, (255, 255, 255), 4, cv2.LINE_AA)

    # Display the image in a window
    cv2.imshow('Counting Window', image)

def displayError(error, imageSize, windowName):
    image = np.zeros(imageSize, np.uint8)
    cv2.putText(image, str(error), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(windowName, image)

 
def timeTaggerThread(queue,sharedMemoryName,sharedMemCounterDataName,sharedMemCorrelationDataName,threadExitCode,terminateTaggerEvent,
                     setTriggerLevelEvent,setGetCounterGraph,setGetCorrelationGraph,setbinNumberEvent,setDelayTimeEvent,
                     getCoincidencesEvent,continuousCaptureMode,singleCaptureMode,
                      measurementChannels,binWidth,binNumber,countingTime,normalizePerSec,numberSamples,
                      selectedChannel,triggerLevel,DelayTime,CloseMem):

    sharedMemory = shared_memory.SharedMemory(name=sharedMemoryName)
    countsBuffer = np.ndarray((4), dtype=np.dtype(int), buffer=sharedMemory.buf)
    
    sharedMemCorrelationData = shared_memory.SharedMemory(name=sharedMemCorrelationDataName)
    correlationData_shm  = np.ndarray((2, binNumber.value), dtype=int, buffer=sharedMemCorrelationData.buf) 
    corrData = np.zeros((2,binNumber.value),dtype=int)
    
    sharedMemCounterData = shared_memory.SharedMemory(name=sharedMemCounterDataName)
    CounterData_shm  = np.ndarray((2, binNumber.value), dtype=int, buffer=sharedMemCounterData.buf) 
    counterData = np.zeros((2,binNumber.value),dtype=int)
    
    taggerDevice = TimeTaggerDevice()
    
    if not taggerDevice.taggerConnected():
        threadExitCode.value = TAGGER_NOT_CONNECTED_CODE
        return
    if measurementChannels[1]== 4:
        taggerDevice.setFiliter(measurementChannels[0])
    coincidencesHistory = []
    contrastHistory = []
    coincidencesAverage = 0
    coincidencesDeviation = 0
    contrastAverage = 0
    contrastDeviation = 0

    image = np.zeros((IMAGE_Y, IMAGE_X, 3), np.uint8)

    continuousCaptureMode.set()
    # setTriggerLevelEvent.set()
    setDelayTimeEvent.set()
    # singleCaptureMode.set()

    while not terminateTaggerEvent.is_set():
        cv2.imshow('Counting Window', image)
        if continuousCaptureMode.is_set():
            # queue.put("test1")
            counts = taggerDevice.getCoincidences(measurementChannels,
                                        binWidth.value,
                                        countingTime.value,
                                        normalizePerSec.value)
            
            channel1Counts=counts[0]
            channel2Counts=counts[1]
            coincidences=counts[2]
            contrast=counts[3]
            # channel1Counts, channel2Counts, coincidences, contrast = counts

            coincidencesHistory.append(coincidences)
            contrastHistory.append(contrast)
            if len(coincidencesHistory) > numberSamples.value:
                coincidencesHistory.pop(0)
                contrastHistory.pop(0)
                
                coincidencesAverage = np.mean(coincidencesHistory)
                coincidencesDeviation = int(np.std(coincidencesHistory))
                contrastAverage = np.mean(contrastHistory)
                contrastDeviation = int(np.std(contrastHistory))
            displayCounts(channel1Counts,channel2Counts,coincidences,
                        coincidencesAverage,coincidencesDeviation,
                        contrast,contrastAverage,contrastDeviation)
        elif singleCaptureMode.is_set():
            if getCoincidencesEvent.is_set():
                counts = taggerDevice.getCoincidences(measurementChannels,
                                                      binWidth.value,
                                                      countingTime.value,
                                                      normalizePerSec.value)
                np.copyto(countsBuffer, counts)
                getCoincidencesEvent.clear()
                
        if setGetCorrelationGraph.is_set():
            # taggerDevice.setDelayTime(measurementChannels[0], DelayTime.value)
            # taggerDevice.setInputDelay(measurementChannels[0], delay.value)
            corrData[0,:],corrData[1,:] = taggerDevice.getCorrelations(measurementChannels,
                                                      binWidth.value,
                                                      binNumber.value,
                                                      countingTime.value)
            np.copyto(correlationData_shm, corrData)
            setGetCorrelationGraph.clear()
        
        if setGetCounterGraph.is_set():
            # taggerDevice.setDelayTime(measurementChannels[0], DelayTime.value)
            # taggerDevice.setInputDelay(measurementChannels[0], delay.value)
            clearbuffer=True
            counterData[0,:],counterData[1,:] = taggerDevice.getCounts(clearbuffer,selectedChannel.value,
                                                      binWidth.value,
                                                      binNumber.value,
                                                      countingTime.value)
            np.copyto(CounterData_shm, counterData)
            setGetCounterGraph.clear()
            
        # Set Properties values are below here the stuff above is more measurement bases    
        if setTriggerLevelEvent.is_set():
            taggerDevice.setTriggerLevel(selectedChannel.value, triggerLevel.value)
            triggerLevel.value = taggerDevice.getTriggerLevel(selectedChannel.value)
            setTriggerLevelEvent.clear()
            
        if setDelayTimeEvent.is_set():
            taggerDevice.setDelayTime(measurementChannels[selectedChannel.value], DelayTime[selectedChannel.value])
            DelayTime[selectedChannel.value]=taggerDevice.getDelayTime(measurementChannels[selectedChannel.value])
            setDelayTimeEvent.clear()
            
        if setbinNumberEvent.is_set():
            if(CloseMem.value==True):
                sharedMemCorrelationData.close()   
                del correlationData_shm
                del corrData
                sharedMemCounterData.close()   
                del CounterData_shm
                del counterData
                CloseMem.value=False
            else:  
            # make new memory space but use the same name so the thread doesnt need to be restarted
            # The memory has been unlinked outside the thread so it just has to be closed inside the thread
            # the "Thing" that is linking the memory together is the sharedMemCorrelationDataname they are the same
            # inside and outside the thread. 
                sharedMemCorrelationData = shared_memory.SharedMemory(name=sharedMemCorrelationDataName)
                correlationData_shm  = np.ndarray((2, binNumber.value), dtype=int, buffer=sharedMemCorrelationData.buf) 
                corrData = np.zeros((2,binNumber.value),dtype=int) 
                sharedMemCounterData = shared_memory.SharedMemory(name=sharedMemCounterDataName)
                CounterData_shm  = np.ndarray((2, binNumber.value), dtype=int, buffer=sharedMemCounterData.buf) 
                counterData = np.zeros((2,binNumber.value),dtype=int)         
            setbinNumberEvent.clear()
       
            

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    sharedMemory.close()
    sharedMemCorrelationData.close()
    sharedMemCounterData.close()
    
    del taggerDevice

    threadExitCode.value = SUCCESS_CODE
    

class TimeTaggerInterface:
    def __init__(self,channels):
        self.threadExitCode = multiprocessing.Value('i', UNKNOWN_ERROR_CODE)
        self.binWidth = multiprocessing.Value('i', 250) # in ps int(1e-9 *1e12)
        self.binNumber = multiprocessing.Value('i', 1000) # number of bins
        self.countingTime = multiprocessing.Value('f', 0.05 *1e12)  # in ps
        self.normalizePerSec = multiprocessing.Value('i', 1)
        self.numberSamples = multiprocessing.Value('i', 20)
        self.measurementChannels = multiprocessing.Array('i', channels)
        self.selectedChannel = multiprocessing.Value('i')
        self.triggerLevel = multiprocessing.Value('f',0.226)
        self.CloseMem = multiprocessing.Value('b',False)
        self.DelayTime = multiprocessing.Array('i', [2150-586-6,6200])

#         tagger.setInputDelay(1, 2150-586-6)
# tagger.setInputDelay(2, 1800-317+12)
# tagger.setInputDelay(3, 1325-302+3)
        
        # self.DelayTime = multiprocessing.Value('i',640)
        
        
        
        
        self.terminateTaggerEvent = multiprocessing.Event()
        self.setTriggerLevelEvent = multiprocessing.Event()
        self.getCoincidencesEvent = multiprocessing.Event()
        self.continuousCaptureMode = multiprocessing.Event()
        self.singleCaptureMode = multiprocessing.Event()
        self.setGetCounterGraph=multiprocessing.Event()
        self.setGetCorrelationGraph=multiprocessing.Event()
        self.setbinNumberEvent=multiprocessing.Event()
        self.setDelayTimeEvent=multiprocessing.Event()
        
        # self.setbinNumberEvent_closeMem=multiprocessing.Event()
        
        self.queue=multiprocessing.Queue()
        
        sharedMemorySize = int(4*np.dtype(int).itemsize)
        self.sharedMemory = shared_memory.SharedMemory(create=True,size=sharedMemorySize)
        self.sharedMemoryName = self.sharedMemory.name
        self.sharedMemoryData = np.ndarray((4), dtype=np.dtype(int), buffer=self.sharedMemory.buf)
        
        self.sharedMemCorrelationData = shared_memory.SharedMemory(create=True, size=int(2* self.binNumber.value * np.dtype(int).itemsize))
        self.sharedMemCorrelationData_arr  = np.ndarray((2, self.binNumber.value), dtype=int, buffer=self.sharedMemCorrelationData.buf)  
        self.sharedMemCorrelationDataName=self.sharedMemCorrelationData.name

        self.sharedMemCounterData = shared_memory.SharedMemory(create=True, size=int(2* self.binNumber.value * np.dtype(int).itemsize))
        self.sharedMemCounterData_arr  = np.ndarray((2, self.binNumber.value), dtype=int, buffer=self.sharedMemCounterData.buf)  
        self.sharedMemCounterDataName=self.sharedMemCounterData.name
        
        print(len(self.measurementChannels ))
        self.timeTaggerThread = self.startTimeTaggerThread()
        
        for ichannel in range(len(self.measurementChannels)):
            self.setTriggerLevel(self.measurementChannels[ichannel],self.triggerLevel.value)
        
    def __del__(self):
        self.terminateTaggerEvent.set()
        self.sharedMemory.close()
        self.sharedMemory.unlink()
        self.sharedMemCorrelationData.close()
        self.sharedMemCorrelationData.unlink()
        self.sharedMemCounterData.close()
        self.sharedMemCounterData.unlink()
        
        
        
    def setTriggerLevel(self, channel, level):
        self.selectedChannel.value = channel
        self.triggerLevel.value = level
        
        self.setTriggerLevelEvent.set()
        while self.setTriggerLevelEvent.is_set():
            time.sleep(WAIT_FOR_THREAD_TIME)
        
        print("New trigger level:", self.triggerLevel.value)

    def setBinWidth(self, binWidth):
        self.binWidth.value = binWidth

    def setCountingTime(self, countingTime):
        self.countingTime.value = countingTime
        
    def setDelayTime(self,channel, DelayTime):
        self.selectedChannel.value
        tempvalue = [i for i in range(len(self.measurementChannels)) if self.measurementChannels[i] == channel]
        print(tempvalue)
        self.selectedChannel.value=tempvalue[0]
        # self.DelayTime.value = DelayTime
        self.DelayTime[self.selectedChannel.value] = DelayTime
        print(self.DelayTime[self.selectedChannel.value])
        
        self.setDelayTimeEvent.set()
        while self.setDelayTimeEvent.is_set():
            time.sleep(WAIT_FOR_THREAD_TIME)
        print(self.DelayTime[self.selectedChannel.value])
        
        
        
    def setBinNumber(self, BinNumber):
        self.binNumber.value=BinNumber
        # when you chnage the bin size you need to unlik and close the memory of the correlationData as it need to chnage size you can then make a new memory space
        self.sharedMemCorrelationData.close()
        self.sharedMemCounterData.close()
        
        
        # Have to now close the memory on the thread so that the unlink will actually release the memory 
        self.CloseMem.value=True
        self.setbinNumberEvent.set()
        while self.setbinNumberEvent.is_set():
            time.sleep(WAIT_FOR_THREAD_TIME)
        self.CloseMem.value=False
        # unlink outside the thread but close inside the thread
        self.sharedMemCorrelationData.unlink() 
        self.sharedMemCounterData.unlink() 
        
        # make new memory space but use the same name so the thread doesnt need to be restarted
        self.sharedMemCorrelationData = shared_memory.SharedMemory(name=self.sharedMemCorrelationDataName,create=True, size=int(2* self.binNumber.value * np.dtype(int).itemsize))
        self.sharedMemCorrelationData_arr  = np.ndarray((2, self.binNumber.value), dtype=int, buffer=self.sharedMemCorrelationData.buf) 
         
        self.sharedMemCounterData = shared_memory.SharedMemory(name=self.sharedMemCounterDataName,create=True, size=int(2* self.binNumber.value * np.dtype(int).itemsize))
        self.sharedMemCounterData_arr  = np.ndarray((2, self.binNumber.value), dtype=int, buffer=self.sharedMemCounterData.buf) 
        # self.sharedMemCorrelationDataName=self.sharedMemCorrelationData.name
        self.setbinNumberEvent.set()
        while self.setbinNumberEvent.is_set():
            time.sleep(WAIT_FOR_THREAD_TIME)
        
    # def setNumberSamples(self, numberSamples):
    #     self.numberSamples.value = numberSamples
    def getCorrelatinoGraph(self):
        self.setGetCorrelationGraph.set()
        while self.setGetCorrelationGraph.is_set():
            time.sleep(WAIT_FOR_THREAD_TIME)
        
        return self.sharedMemCorrelationData_arr
    
    def getCounterGraph(self):
        self.setGetCounterGraph.set()
        while self.setGetCounterGraph.is_set():
            time.sleep(WAIT_FOR_THREAD_TIME)
        
        return self.sharedMemCounterData_arr
    
    def getCoincidences(self, normalisePerSec=False):
        if normalisePerSec:
            self.normalizePerSec.value = 1

        self.getCoincidencesEvent.set()
        while self.getCoincidencesEvent.is_set():
            time.sleep(WAIT_FOR_THREAD_TIME)
        
        return self.sharedMemoryData
    
    def setSingleCaptureMode(self):
        self.continuousCaptureMode.clear()
        self.singleCaptureMode.set()
        self.normalizePerSec.value = 0
        
    def setContinuousCaptureMode(self):
        self.singleCaptureMode.clear()
        self.continuousCaptureMode.set()
        self.normalizePerSec.value = 1
        
    def startTimeTaggerThread(self):
        threadArgs = (
             self.queue,
            self.sharedMemoryName,
            self.sharedMemCounterDataName,
            self.sharedMemCorrelationDataName,
            self.threadExitCode,
            self.terminateTaggerEvent,
            self.setTriggerLevelEvent,
            self.setGetCorrelationGraph,
            self.setGetCounterGraph,
            self.setbinNumberEvent,
            self.setDelayTimeEvent,
            self.getCoincidencesEvent,
            self.continuousCaptureMode,
            self.singleCaptureMode,
            self.measurementChannels,
            self.binWidth,
            self.binNumber,
            self.countingTime,
            self.normalizePerSec,
            self.numberSamples,
            self.selectedChannel,
            self.triggerLevel,
            self.DelayTime,
            self.CloseMem
        )
        
        process = multiprocessing.Process(target=timeTaggerThread, args=threadArgs)
        process.start()
        # print(process)
        
        return process
    
    def isThreadRunning(self):
        if self.timeTaggerThread.is_alive():
            print("Time tagger thread is running")
            return True
        else:
            print("Thread not running:",EXIT_MESSAGE[self.threadExitCode.value])
            return False
    
class TimeTaggerDevice:
    def __init__(self):
        self.tagger = None
        try:
            self.tagger = TimeTagger.createTimeTagger()
        except RuntimeError as e:
            print(e)
            return
        
    def __del__(self):
        if self.tagger is None:
            return
        
        TimeTagger.freeTimeTagger(tagger=self.tagger)
        
    def setFiliter(self,channel):
        self.tagger.setConditionalFilter(trigger=[channel], filtered=[4])
    
    def taggerConnected(self):
        return self.tagger is not None
    
    def setDelayTime(self,channel,DelayTime):
        self.tagger.setInputDelay(channel, DelayTime)
        
    def getDelayTime(self,channel):
        Delay=self.tagger.getInputDelay(channel)
        return Delay
    
    def setTriggerLevel(self, channel, level):
        self.tagger.setTriggerLevel(channel, level)
        
    def getTriggerLevel(self, channel):
        return self.tagger.getTriggerLevel(channel)
    
    def getCoincidences(self, measurementChannels, binWidth, countingTime, normalizePerSecond=0):
        coincidenceMeasurement = TimeTagger.Coincidence(
            tagger=self.tagger, 
            channels=measurementChannels,
            coincidenceWindow=binWidth,
            timestamp=TimeTagger.CoincidenceTimestamp.Last
        )
        coincidenceChannel = coincidenceMeasurement.getChannel()
        
        coincidenceRate = TimeTagger.Countrate(tagger=self.tagger, channels=[*measurementChannels, coincidenceChannel])
        coincidenceRate.startFor(countingTime)
        coincidenceRate.waitUntilFinished()
        
        channel1Counts,channel2Counts,coincidences = coincidenceRate.getCountsTotal()

        if normalizePerSecond:
            countingTimeInSec = countingTime*1e-12

            channel1CountsPerSec = int(channel1Counts/countingTimeInSec)
            channel2CountsPerSec = int(channel2Counts/countingTimeInSec)
            coincidencesPerSec = int(coincidences/countingTimeInSec)

        if(channel1Counts != 0 and channel2Counts !=0):
            accidentals = channel1Counts*channel2Counts*2*(binWidth*1e-12)
            contrast = int((coincidences - accidentals)/accidentals)
        else:
            contrast=0

        if normalizePerSecond:
            counts = np.asarray([channel1CountsPerSec, channel2CountsPerSec, coincidencesPerSec, contrast])
        else:
            counts = np.asarray([channel1Counts, channel2Counts, coincidences, contrast])
        
        return counts
    
    def getCorrelations(self, measurementChannels, binWidth, n_bin,countingTime):
        
        correlation = TimeTagger.Correlation(tagger=self.tagger,
                                     channel_1=measurementChannels[0],
                                     channel_2=measurementChannels[1],
                                     binwidth=binWidth,
                                     n_bins=n_bin)
        correlation.startFor(capture_duration=int(countingTime))
        correlation.waitUntilFinished()
        correlation_data = np.asarray(correlation.getData())
        correlationTimebins_data=np.asarray(correlation.getIndex())
        return correlationTimebins_data,correlation_data
    
    def getCounts(self,clearbuffer, measurementChannel, binWidth, n_bin,countingTime):
        
        counter = TimeTagger.Counter(tagger=self.tagger,
                                     channels=[measurementChannel],
                                     binwidth=binWidth,
                                     n_values=n_bin)
        if (clearbuffer==True):
            counter.clear()
        counter.startFor(capture_duration=int(countingTime))
        counter.waitUntilFinished()
        Couter_data = np.asarray(counter.getData())
        CounterTimebins_data=np.asarray(counter.getIndex())
        return CounterTimebins_data,Couter_data[0,:]
    

