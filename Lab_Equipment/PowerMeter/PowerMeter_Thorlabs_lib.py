from Lab_Equipment.Config import config
import pyvisa
# from ThorlabsPM100 import ThorlabsPM100 # go and look at documentation to see the other functions that can be used to change things on the power meter. You need to pip install this
import ThorlabsPM100
import numpy as np

# OK... so... there is some crazy bull shit with the drivers with the PM100D, 
#if you install the thorlabs software the drivers are set to TLM drivers (these are the new drivers) the drivers need to be NI-VISA driver.
#To switch the drivers there is a some software located in C:\Program Files (x86)\Thorlabs\OPM\Tools\DriverSwitcher called Thorlabs.PMDriverSwitcher.exe. You can switch the drivers there.
#NOTE the USB codes are USB[board]::vendorID::productID::serialNumber::INSTR example 'USB::0x1313::0x8078::P0024994::INSTR'
class PowerMeterObj:
    def __init__(self, deviceName='',wavelength=925,AvgCount=1,Units="W"):
        # super().__init__()  # Initialize the base class
        self.deviceName=deviceName
        self.wavelength=wavelength # this is in nm 
        self.AvgCount=AvgCount
        self.Units=Units
        # open a Resource Manger to look for the power meter
        rm = pyvisa.ResourceManager()# This just looks at all the USB stuff that is on the computer
        # If the user has no idea the ID of the power meter the class will give a list of the connected device so the user can re-initialise with to correct ID 
        if (deviceName==''):
            print("No devices given the list below shows connected device please work out which one is the power meter and re-initialise the class with the string")
            print("USB codes are USB[board]::vendorID::productID::serialNumber::INSTR example 'USB::0x1313::0x8078::P0024994::INSTR' \n")
            # rm.list_resources()# this will print out all the USB connected device you need to manually work out what is what. Unplug and re-run to see what is on the list and what isn't
            print(rm.list_resources())
            return

        # connect to the device
        self.inst1 = rm.open_resource(str(self.deviceName),timeout=100000000)# this gets the specific USB deivce that you want to access

        self.power_meter = ThorlabsPM100.ThorlabsPM100(inst=self.inst1)# this invokes the Thorlabs python lib that hides the tedious serial writes to the powermeter
     
        #set the average counts for the device 
        self.SetAverageMeasure(self.AvgCount)
        
        #set the wavelength for the device 
        self.SetWaveLength(self.wavelength)
        
        #set the units for the device 
        self.SetUnits(self.Units)
        
        # Get a power Measruement
        self.pwr=self.GetPower()
        
        
    def __del__(self):
        # stop the connection to the power meter
        self.inst1.close()
        
    def SetAverageMeasure(self,AvgCount):
        self.AvgCount=AvgCount
        self.power_meter.sense.average.count=self.AvgCount
        
    def SetWaveLength(self,Wavelength):
        self.wavelength=Wavelength
        self.power_meter.sense.correction.wavelength=self.wavelength
        print("Wavelength set to "+str(self.wavelength)+' nm')
        # The two lines below do the same thing as the one line above it is just that the lines below do a specific serial write to the power meter
        # the specific codes like SENS:CORR:WAV are specified in the ThorlabsPM100 library
        # wavelength=1565
        # power_meter.inst.write('SENS:CORR:WAV %f' %wavelength)  
    
    def SetUnits(self,Units):
        self.Units=Units
        self.power_meter.sense.power.dc.unit=self.Units
        #NOTE when ever you change the units you need to call configure.X.X(). This is the same with if you wanted to measure a different value like voltage or current.
        self.power_meter.configure.scalar.power()
        print("Units set to "+str(self.Units))
        
        
    def GetPower(self):# this will get the currently configured measure value.
        self.pwr=self.power_meter.read
        return self.pwr
        
