import numpy as np
import numexpr as ne #ne.set_num_threads(16) # I am not useing large set of arrays so 8 seems the best DANIEL not sure what this commet is talking about 
import time
import sys
import pickle
# import pathlib

#If I do not do from folder.fullscreenqt import module, super(). will fail when LCOS called from other module -- I do not know what the F*** is happeing
# from spatial_light_modulator.fullscreenqt import FullscreenWindow
from Lab_Equipment.Config import config 
from  Lab_Equipment.SLM.fullscreenqt import FullscreenWindow
import Lab_Equipment.ZernikeModule.ZernikeModule as zernMod
class MaskProperties:
    def __init__(self, zernike,zernikeEnable, MaskCmplx,maskPatternEnable,MaskPlusZern,maskEnabled, att_enabled, attWeight, center):
        self.zernike = zernike
        self.zernikeEnable=zernikeEnable
        self.MaskCmplx = MaskCmplx # this is the pattern without zernike essentially the pase of the masks you want to put on the SLM
        self.maskPatternEnable = maskPatternEnable
        self.MaskPlusZern = MaskPlusZern # this is mask that is actually displayed on SLM
        self.att_enabled = att_enabled
        self.attWeight = attWeight
        self.center = center
        self.maskEnabled=maskEnabled
class GlobalProperties():
    def __init__(self,rgbChannel,rgbChannelIdx,slmEnable, zernikeEnable,polEnabled, maskPatternEnable,RefreshTime):
        self.rgbChannel=rgbChannel
        self.rgbChannelIdx=rgbChannelIdx
        self.zernikeEnable=zernikeEnable
        self.polEnabled=polEnabled
        self.maskPatternEnable=maskPatternEnable
        self.slmEnable=slmEnable
        self.RefreshTime= RefreshTime
class LCOS(FullscreenWindow):
    def __init__(self, screen = 1, ActiveRGBChannels=["Red"],pixel_size = 9.2e-6, aperture_diameter = 7.5e-3, Mask=np.ones((1,1,960,960),dtype=np.csingle),RefreshTime=600e-3, MODELAB_COMPATIBILITY = True, **kwargs):
        """_summary_
        Args:
            screen (int, optional): index of the monitor to be used to display the FULL MASKS. Defaults to 1.
            pixel_size (_type_, optional): pixel pitch of the SLM. Defaults to 9.2e-6.
            aperture_diameter (_type_, optional): Diemater of the aperture to used to crop the masks in m. Defaults to 7.5e-3.
            mask_size (tuple, optional): size of the independed mask provided (zernikes H and V, patterns ...). Defaults to (960,960).
            MODELAB_COMPATIBILITY (bool) : Use when using zernikes optimized in Modelab.
                                            It only affects to masks centers. It applies some offsets to the provided centers(modelab centers)
                                            to be able to center the mask at the same spot as in Modelab

            **kwargs: 
                zernike_H (array), zernike_V (array), pattern_H (array), pattern_V(array), HmaskCenter ([x,y]), VmaskCenter([x,y]), 
                polselect('H', 'V' or 'HV'), zernikesEnabled (0 or 1), patternEnabled (0 or 1)
        """
        ####################################
        # NOTE when address zernikes in code these are the index values that should be used 
        # to address the specific values
        ####################################
        print("""
        Zern Coefs:
        0:  piston    ( 0,  0 )
        1:  Tiltx     (-1,  1 )
        2:  Tilty     ( 1,  1 )
        3:  Astigx    (-2,  2 )
        4:  Defocus   ( 0,  2 )
        5:  Astigy    ( 2,  2 )
        6:  Trefilx   (-3,  3 )
        7:  Comax     (-1,  3 )
        8:  Comay     ( 1,  3 )
        9:  Trefoily  ( 3,  3 )
        12:  Spherical ( 0,  4 )
        """)
        self.display  = screen
        
        super().__init__(screen = self.display) #This init the screen
        
        self.screen_data = self.getBuffer() #(Y,X,RGB)
        self.LCOSsize = self.screen_data.shape[0:2]
        self.pixel_size  = pixel_size
        self.aperture_diameter = aperture_diameter
        self.slmWidth=self.LCOSsize[1]
        self.slmHeigth=self.LCOSsize[0]
        self.modeCount_step=1
        self.modeCount_start=0
        
        self.apertureApplied=np.ones(self.LCOSsize)
        
        #INIT OBJECT MEM SPACE
        # att_period = 16

        self.FullScreenBufferIntimediate_int = np.zeros(self.LCOSsize,dtype=np.uint8)
        self.FullScreenBuffer_int = np.zeros(self.LCOSsize,dtype=np.uint8)
        
        self.FullScreenBufferIntimediate_cmplx = np.zeros(self.LCOSsize,dtype=complex)
        self.FullScreenBuffer_cmplx = np.zeros(self.LCOSsize,dtype=complex)
        
        
        # self.LCOS_array_H = np.zeros(self.LCOSsize,dtype=float) #See performance with floats64
        # self.LCOS_array_V = np. copy(self.LCOS_array_H)
        # self.LCOS_array = np.copy(self.LCOS_array_H)
        # self.LCOS_Screen_temp = np.zeros(self.LCOSsize,dtype=float)

        # self.LCOS_array_cmplx = np.zeros(self.LCOSsize,dtype=complex) #See performance with floats64
        # self.LCOS_Screen_temp_cmplx = np.zeros(self.LCOSsize,dtype=complex)

        self.AllMaskProperties={}#empty dictionary
        self.GLobProps={}
        self.channelCount=len(ActiveRGBChannels)
        self.ActiveRGBChannels=ActiveRGBChannels
        for ichannel in range(self.channelCount):
            if(( self.ActiveRGBChannels[ichannel]=="Red" or self.ActiveRGBChannels[ichannel]=="Green" or  self.ActiveRGBChannels[ichannel]=="Blue") == False):
                print("You haven't input a channel colour correctly. It needs to be either Red, Green or Blue with that exact spelling")
                return 
            
        self.RefreshTimetemp=RefreshTime
        
        # #Apertures masks
        self.ap_H = None
        self.ap_V = None
        self.ap = None
        # # self.defineCenters()
        # # NOTE I think this will need to be implemented but for the time being I just want to see thing working so I am going to ignore it for the moment
        # # self.calcApertures() #This method should be call each time, centers aperture diameter and pixel size changes
                    
        # # Added masks 
        # self.Hmask = None # Result of adding Hmask_specs
        # self.Vmask = None 
        
        self.DisplayedPhaseMask = None #After processing the 2 independend masks HV with other parameters and centers
        self.apertureApplied = None # The aperture filter applied
        self.DisplayedLevelMask = None # Final mask from 0 to 255
        
        self.refreshfreq = 0
        for ichannel in range(self.channelCount):
            self.Initialise_SLM_Mask(self.ActiveRGBChannels[ichannel],Mask)

        # self.attenuationPattern = self.binarycheckboard(self.masksize[0],self.masksize[1], spatial_frequency = 1/att_period,scale = 1, offset= True, pol='HV') #using HV to get the whole mask

        
    def Initialise_SLM_Mask(self,channel,Mask,LoadMASKSetting=False,MaskSettingFilePrefixName=''):
        dims=Mask.shape
        self.modeCount= dims[0]
        # mode_count = 21
        self.MaskCount = dims[1]
        self.masksize = dims[2:]# this is sometime refered to in other files as planeCount due to context of MPLC
        att_period = 16
        self.attenuationPattern = self.binarycheckboard(self.masksize[0],self.masksize[1], spatial_frequency = 1/att_period,scale = 1, offset= True, pol='HV') #using HV to get the whole mask

        
        # Due to the fact that there are no mask properties loaded when the class is initally loaded I set the centers to all be equally spaced along the SLM
        # This might not be the best why to initalies all this but it works and at least anyone suing the code will see all the masks along the SLM
        # once the masks have been initalised the setting from a masksetting file can be implemented
        self.MaskSizeDefult=(self.LCOSsize[1]//self.MaskCount)
        HmaskCenter_temp=np.ones((self.MaskCount,2),dtype=int)
        VmaskCenter_temp=np.ones((self.MaskCount,2),dtype=int)
        HmaskCenter_temp[:,1]=np.arange((self.MaskSizeDefult)//2-1,self.LCOSsize[1]-self.MaskSizeDefult//2,self.MaskSizeDefult)
        HmaskCenter_temp[:,0]=HmaskCenter_temp[:,0]*self.LCOSsize[0]//4
        VmaskCenter_temp[:,1]=np.arange((self.MaskSizeDefult)//2-1,self.LCOSsize[1]-self.MaskSizeDefult//2,self.MaskSizeDefult)
        VmaskCenter_temp[:,0]=VmaskCenter_temp[:,0]*((self.LCOSsize[0]-1)-self.LCOSsize[0]//4)
            
        
        # for ichannel in range(self.channelCount):
        if( channel=="Red"):
            rgbChannelIdx=0
        elif (channel=="Green"):
            rgbChannelIdx=1
        elif( channel=="Blue"):
            rgbChannelIdx=2

        H_MaskSetProperties = []
        V_MaskSetProperties = []
        for imask in range(self.MaskCount):
            zernike = zernMod.Zernikes(max_zernike_radial_number=4, Nx=self.masksize[0], Ny=self.masksize[1],load_modelab=False)
            maskProp = MaskProperties(
                zernike=zernike,
                zernikeEnable=1,
                # maskPattern=np.zeros((self.modeCount,self.masksize[0],self.masksize[1]),np.float32),
                MaskCmplx=np.zeros((self.modeCount,self.masksize[0],self.masksize[1]),complex),

                maskPatternEnable=1,
                MaskPlusZern=np.zeros((self.modeCount,self.masksize[0],self.masksize[1]),complex),
                # Mask=np.zeros((self.modeCount,self.masksize[0],self.masksize[1]),np.float32),

                maskEnabled=True,
                att_enabled=0, 
                attWeight=0, 
                center=[HmaskCenter_temp[imask,0],HmaskCenter_temp[imask,1]]
                )
            H_MaskSetProperties.append(maskProp)
            zernike = zernMod.Zernikes(max_zernike_radial_number=4, Nx=self.masksize[0], Ny=self.masksize[1],load_modelab=False)
            maskProp = MaskProperties(
                zernike=zernike,
                zernikeEnable=1, 
                # maskPattern=np.zeros((self.modeCount,self.masksize[0],self.masksize[1]),np.float32),
                MaskCmplx=np.zeros((self.modeCount,self.masksize[0],self.masksize[1]),complex),

                maskPatternEnable=1, 
                # Mask=np.zeros((self.modeCount,self.masksize[0],self.masksize[1]),np.float32),
                MaskPlusZern=np.zeros((self.modeCount,self.masksize[0],self.masksize[1]),complex),

                maskEnabled=True,
                att_enabled=0, 
                attWeight=0, 
                center=[VmaskCenter_temp[imask,0],VmaskCenter_temp[imask,1]]
                )
            V_MaskSetProperties.append(maskProp)        

        ############# IMPORTANT ATRIBUTES - THEY CONTROL THE OBJECT #################################      
        AllMaskPropertiesSingleChannel = {'H': H_MaskSetProperties, 'V': V_MaskSetProperties} #After the object is created, modifying the parameters in this dictionary and runing setmask will update the LCOS mask
        self.AllMaskProperties[ channel]=AllMaskPropertiesSingleChannel
        
        #Control flags to add the masks
        GLobPropsTemp=GlobalProperties(
                rgbChannel= channel,
                rgbChannelIdx=rgbChannelIdx,
                zernikeEnable=[True,True],
                polEnabled=[True,True],
                maskPatternEnable=[True,True],
                slmEnable=True,
                RefreshTime=self.RefreshTimetemp
            )
        
        self.GLobProps[channel]= GLobPropsTemp

        # for ichannel in range(self.channelCount):
        self.setMaskArray(channel, Mask)
        self.setmask(channel)
        if (LoadMASKSetting):
            self.LoadMaskProperties(MaskSettingFilePrefixName)


    def saveMaskProperties(self,filenamePrefix=''):

        # Open a file for writing. The 'wb' argument indicates 'write binary' mode.
        for channel  in self.ActiveRGBChannels:
            xcentersH=np.zeros(self.MaskCount,dtype=int)
            ycentersH=np.zeros(self.MaskCount,dtype=int)
            xcentersV=np.zeros(self.MaskCount,dtype=int)
            ycentersV=np.zeros(self.MaskCount,dtype=int)
            zernikeCount=self.AllMaskProperties[channel]["H"][0].zernike.zernCount
            zernikesPropsH=np.zeros((self.MaskCount,zernikeCount))
            zernikesPropsV=np.zeros((self.MaskCount,zernikeCount))
            # NOTE need to move all the variables into arrays as the imask index cant be sliced on the AllMaskProperties which is a little annoying 
            # but that is the price you pay for dict object
            for imask in range(self.MaskCount):
                xcentersH[imask]=self.AllMaskProperties[channel]["H"][imask].center[0]
                ycentersH[imask]=self.AllMaskProperties[channel]["H"][imask].center[1]
                xcentersV[imask]=self.AllMaskProperties[channel]["V"][imask].center[0]
                ycentersV[imask]=self.AllMaskProperties[channel]["V"][imask].center[1]
                zernikesPropsH[imask,:]=self.AllMaskProperties[channel]["H"][imask].zernike.zern_coefs
                zernikesPropsV[imask,:]=self.AllMaskProperties[channel]["V"][imask].zernike.zern_coefs
        
            GlobalProp=self.GLobProps[channel]
            with open(config.SLM_LIB_PATH+'Data'+config.SLASH+'MASKProperties_'+channel+filenamePrefix+'.pkl', 'wb') as file:
                # Use pickle to dump the variables into the file
                pickle.dump((xcentersH,ycentersH,xcentersV,ycentersV,zernikesPropsH,zernikesPropsV,GlobalProp),file)
                
    def LoadMaskProperties(self,filenamePrefix=''):
        # Open a file for writing. The 'wb' argument indicates 'write binary' mode.
        for channel  in self.ActiveRGBChannels:
            with open(config.SLM_LIB_PATH+'Data'+config.SLASH+'MASKProperties_'+channel+filenamePrefix+'.pkl', 'rb') as file:
                # Use pickle to load the variables from the file
                xcentersH, ycentersH,xcentersV, ycentersV, zernikesPropsH,zernikesPropsV,GlobalProp = pickle.load(file)
            
                for imask in range(len(xcentersH)):
                    self.AllMaskProperties[channel]["H"][imask].center[0]=xcentersH[imask]
                    self.AllMaskProperties[channel]["H"][imask].center[1]=ycentersH[imask]
                    self.AllMaskProperties[channel]["V"][imask].center[0]=xcentersV[imask]
                    self.AllMaskProperties[channel]["V"][imask].center[1]=ycentersV[imask]
                    self.AllMaskProperties[channel]["H"][imask].zernike.zern_coefs=zernikesPropsH[imask,:]
                    self.AllMaskProperties[channel]["V"][imask].zernike.zern_coefs=zernikesPropsV[imask,:]
                
                self.GLobProps[channel]=GlobalProp
        for channel  in self.ActiveRGBChannels:
            if(channel=="Red"):
                self.setmask(channel,0)
            if(channel=="Green"):
                self.setmask(channel,0)

    def ResetAllZernikesToZero(self,channel):
        for ipol in ["H","V"]:
            for imask in range(self.MaskCount):
                for izern in range(self.AllMaskProperties[channel]["H"][0].zernike.zernCount):
                    self.AllMaskProperties[channel][ipol][imask].zernike.zern_coefs[izern]=0
    def UpdateZernike(self,channel):
        imask=0
        pol='H'
        ipol=0
        for icounter in range(self.MaskCount*2):
            if icounter==self.MaskCount:
                pol='V'
                ipol=1
                imask=0
            self.AllMaskProperties[channel][pol][imask].zernike.make_zernike_fields()
            imask=imask+1

    
        


    def ApplyZernikesToSingleMask(self,channel,MaskCmplx,imask,pol,imode=0):
        if pol == "H":
            ipol=0
        if pol == "V":
            ipol=1
        a = np.angle(self.AllMaskProperties[channel][pol][imask].zernike.field) #This should come from -pi to pi
        a1 = self.AllMaskProperties[channel][pol][imask].zernikeEnable
        if self.GLobProps[channel].zernikeEnable[ipol]==False:
            a1 = 0
        # a1 = self.zernikesEnabled
        # b = self.AllMaskProperties[pol][imask].maskPattern[imode,:,:] #This shohould come as -pi to pi ie it is and angle of the mask
        b = MaskCmplx #This shohould come as -pi to pi ie it is and angle of the mask
        b1 = self.AllMaskProperties[channel][pol][imask].maskPatternEnable
        if self.GLobProps[channel].maskPatternEnable[ipol]==False:
            b = 1
        
        att_phi_H = self.CalcAttPhase(self.AllMaskProperties[channel][pol][imask].attWeight)
        c = self.attenuationPattern # array from -0.5pi to 0.5pi for a total of pi phase attenuation
        c1 = self.AllMaskProperties[channel][pol][imask].att_enabled * att_phi_H # attenuation weight should go from -attphi/2 to attphi/2 to avoid pistoning effect 
        # Mask = self.getAngle(ne.evaluate('exp(1j*((a*a1) + (b*b1) + (c*c1)))')) #This wrap the phase from -pi to pi 
     
        Mask = (ne.evaluate('b*exp(1j*((a*a1)  + (c*c1)))')) #This wrap the phase from -pi to pi 
        return Mask
        
    def ApplyZernikesToAllMasks(self,channel,imode=0):
        imask=0
        pol='H'
        ipol=0
        for icounter in range(self.MaskCount*2):
            if icounter==self.MaskCount:
                pol='V'
                ipol=1
                imask=0
            self.AllMaskProperties[channel][pol][imask].MaskPlusZern[imode,:,:]=self.ApplyZernikesToSingleMask(channel,self.AllMaskProperties[channel][pol][imask].MaskCmplx[imode,:,:],imask,pol,imode)
            imask=imask+1

                
   
    
    # These are some old function from originial original code I dont think i need them any more but i will keep here just in case
    # def calcApertures(self):
    #     self.ap_H = self.aperture(self.aperture_diameter,self.Hcenter,self.LCOSsize,self.pixel_size)
    #     self.ap_V = self.aperture(self.aperture_diameter,self.Vcenter,self.LCOSsize,self.pixel_size)
    #     self.ap = np.logical_or(self.ap_H,self.ap_V)

    # def setCenters(self, centerH, centerV):
    #     self.defineCenters()
    #     self.calcApertures()
    # def defineCenters(self,channel):
    #     cH = self.AllMaskProperties[channel]['H'][:].centers
    #     cV =  self.AllMaskProperties[channel]['V'][:].centers                  
    #     self.Hcenter = [cH[:,0] - self.offset_center, cH[:,1] - self.offset_center]
    #     self.Vcenter = [cV[:,0] - self.offset_center, cV[:,1] - self.offset_center]
        
    # def resetAttenuation(self):
    #     for pol in ['H','V']:
    #         self.mask_specs[pol]['att_enabled'] = 0
    #         self.mask_specs[pol]['attWeight'] = 0
  
    def Draw_Single_Mask(self, x_center, y_center, Mask:np.ndarray):
        
        if np.issubdtype(Mask.dtype, np.integer):
            self.FullScreenBufferIntimediate_int.fill(0)
            FullScreenBuffer= self.FullScreenBufferIntimediate_int
        elif np.issubdtype(Mask.dtype, np.complexfloating):
            self.FullScreenBufferIntimediate_cmplx.fill(0+0j)
            FullScreenBuffer= self.FullScreenBufferIntimediate_cmplx
            
        else:
            print("Array is of another type: " +str(Mask.dtype))
            print ("Array must be of int or complex type.")
        self.slmWidth=self.LCOSsize[1]
        self.slmHeigth=self.LCOSsize[0]
        x=x_center-Mask.shape[1]//2
        y=y_center-Mask.shape[0]//2
        # This condtional is for when the mask is off the screen
        if x + Mask.shape[1] < 0 or x > self.slmWidth or y + Mask.shape[0] < 0 or y > self.slmHeigth:
            # return  # Do nothing if the rectangle is outside the space
            if x+ Mask.shape[1]<0:
                x=-Mask.shape[1]
            elif x > self.slmWidth:
                x=self.slmWidth
            if y+ Mask.shape[0]<0:
                y=-Mask.shape[0]
            elif y > self.slmHeigth:
                y=self.slmHeigth   

        # Determine the visible part of the rectangle inside the space
        x_offset = int(max(0, -x))
        y_offset = int(max(0, -y))
        # Determine the visible part of the rectangle inside the space
        x_start =int( max(x, 0))
        y_start = int(max(y, 0))
        x_end = int(min(x + Mask.shape[1], self.slmWidth))
        y_end = int(min(y + Mask.shape[0], self.slmHeigth))

        # Apply the pattern to the appropriate region of the space
        # self.LCOS_Screen_temp[y_start:y_end, x_start:x_end] = Mask[y_offset:y_offset + y_end - y_start, 
        #                                             x_offset:x_offset + x_end - x_start]
        FullScreenBuffer[y_start:y_end, x_start:x_end] = Mask[y_offset:y_offset + y_end - y_start, 
                                                    x_offset:x_offset + x_end - x_start]
        return FullScreenBuffer


    def Draw_All_Masks(self,channel,imode=0):       
        # self.LCOS_array.fill(0)
        self.FullScreenBuffer_cmplx.fill(0)

        pol='H'
        ipol=0
        imask=0
        for icounter in range(self.MaskCount*2):
            if icounter==self.MaskCount:
                pol='V'
                ipol=1
                imask=0
            if(self.GLobProps[channel].polEnabled[ipol]):
                if(self.AllMaskProperties[channel][pol][imask].maskEnabled):
                    x_center=self.AllMaskProperties[channel][pol][imask].center[1]
                    y_center=self.AllMaskProperties[channel][pol][imask].center[0]
                    Mask=(self.AllMaskProperties[channel][pol][imask].MaskPlusZern[imode,:,:])
                    # Mask_grayscale = self.phaseTolevel(Mask )
                    # self.LCOS_Screen_temp =self.Draw_Single_Mask(x_center, y_center, Mask)
                    # self.LCOS_array=self.LCOS_array+self.LCOS_Screen_temp

                    # self.LCOS_Screen_temp_cmplx =self.Draw_Single_Mask(x_center, y_center, Mask)
                    self.FullScreenBuffer_cmplx=self.FullScreenBuffer_cmplx+self.Draw_Single_Mask(x_center, y_center, Mask)
                
            imask=imask+1
        return self.FullScreenBuffer_cmplx
    
    def setMaskArray(self,channel,MASKS):
        dims=MASKS.shape
        if(dims[0]==self.modeCount and dims[1]==self.MaskCount and dims[2:]==self.masksize):
            for imode in range(self.modeCount):
                imask=0
                pol='H'
                for icounter in range(self.MaskCount*2):
                    if icounter==self.MaskCount:
                        pol='V'
                        imask=0
                    # self.AllMaskProperties[channel][pol][imask].maskPattern[imode,:,:]=np.angle(MASKS[imode,imask,:,:])
                    self.AllMaskProperties[channel][pol][imask].MaskCmplx[imode,:,:]=(MASKS[imode,imask,:,:])

                    imask=imask+1
            self.setmask(channel)
        else:
            self.Initialise_SLM_Mask(channel,MASKS)

            
     
    #Takes new patterns in case you only want to update new patterns on top zernikes attenuation etc etc. Otherwise it will take self parameters and build the mask
    def setmask(self,channel="Red",imode=0 ):
        t1 = time.time()
        self.UpdateZernike(channel)
        self.ApplyZernikesToAllMasks(channel,imode) #Update the masks       
        _=self.Draw_All_Masks(channel,imode)
        self.FullScreenBuffer_int=self.convert_phase_to_uint8() # Note if nothing is passed it will use the self.FullScreenBuffer_cmplx array as the array it is going to convert      
        self.Write_To_Display(self.FullScreenBuffer_int,channel)
        etime = time.time() - t1
        self.refreshfreq = 1/etime
    
            
    
    def convert_phase_to_uint8(self,arr=None):
        """
        Convert the phase of each non-zero complex element in a 2D array 
        into a uint8 value in the range 0 to 255.
        
        For each element where the complex value is not 0+0j, the function:
        1. Computes the phase using numexpr's evaluated arctan2 (in radians)
        2. Converts the phase from the [-pi, pi] range to the [0, 255] range
            using the formula:
                scaled_value = round(((angle + pi) / (2*pi)) * 255)
        Elements that are 0+0j are left as 0.
        
        Parameters:
            arr (np.ndarray): A 2D NumPy array of complex numbers.
        
        Returns:
            np.ndarray: A 2D uint8 array with the converted phase values.
        """
        # Preallocate output array, defaulting all values to 0.
        if arr is None:
        # out = np.zeros(arr.shape, dtype=np.uint8)
            self.FullScreenBuffer_int.fill(0) 
            arr_int= self.FullScreenBuffer_int # this is just makeing a alis to arr_int not a copy
            arr=self.FullScreenBuffer_cmplx
        else:
            arr_int = np.zeros(arr.shape, dtype=np.uint8)
        
        
        # Create a mask for entries that are not exactly 0+0j.
        mask = (arr != 0)
        
        if np.any(mask):
            # Extract the real and imaginary parts for the masked entries.
            re = arr.real[mask]
            im = arr.imag[mask]
            
            # Compute the phase angle using numexpr.
            # Note: arctan2 returns values in [-pi, pi].
            angle = ne.evaluate("arctan2(im, re)", local_dict={"im": im, "re": re})
            
            # Convert angle to the range [0, 255]:
            #   scaled_value = ((angle + pi) / (2*pi)) * 255
            # We use np.rint to round to the nearest integer.
            # out[mask] = np.rint((angle + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
            
            # Note: numexpr does not have a rounding function, so we evaluate the expression
            # and then apply np.rint to round to the nearest integer.
            scaled_expr = "((angle + pi) / (2 * pi)) * 255"
            # Evaluate the expression using numexpr.
            scaled = ne.evaluate(scaled_expr, local_dict={"angle": angle, "pi": np.pi})
            # Round and cast to uint8.
            arr_int[mask] = np.rint(scaled).astype(np.uint8)
        
        return arr_int

    #To write into the LCOS
    def Write_To_Display(self, arr_data, channel="Red"):
        channelIdx=self.GLobProps[channel].rgbChannelIdx
        self.screen_data[:,:,channelIdx] = arr_data
        self.update()
        time.sleep(self.GLobProps[channel].RefreshTime)
    
    def LCOS_Clean(self,channel="Red"):
        self.Write_To_Display(np.zeros(self.LCOSsize),channel)
        time.sleep(self.GLobProps[channel].RefreshTime)
        # self.Write_To_Display(np.zeros(self.LCOSsize),self.GLobProps[channel].rgbChannelIdx)
    def setCentersToEqualSpacing(self,channel):
     
        maskCenter=np.ones((2,self.MaskCount,2),dtype=int)
        maskCenter[0,:,1]=np.arange((self.MaskSizeDefult)//2-1,self.LCOSsize[1]-self.MaskSizeDefult//2,self.MaskSizeDefult)
        maskCenter[0,:,0]=maskCenter[0,:,0]*self.LCOSsize[0]//4
        maskCenter[1,:,1]=np.arange((self.MaskSizeDefult)//2-1,self.LCOSsize[1]-self.MaskSizeDefult//2,self.MaskSizeDefult)
        maskCenter[1,:,0]=maskCenter[1,:,0]*((self.LCOSsize[0]-1)-self.LCOSsize[0]//4)
        imask=0
        pol='H'
        ipol=0
        for icounter in range(self.MaskCount*2):
            if icounter==self.MaskCount:
                pol='V'
                ipol=1
                imask=0
            self.AllMaskProperties[channel][pol][imask].center=[maskCenter[ipol,imask,0],maskCenter[ipol,imask,1]]
            imask=imask+1
        self.setmask(channel)
   
    def printCenters(self,channel):
        for imask in range(self.MaskCount):
            print(self.AllMaskProperties[channel]['H'][imask].center)
            # print(self.AllMaskProperties['H'][0].center[1])  
        print('\n')
        for imask in range(self.MaskCount):   
            print(self.AllMaskProperties[channel]['V'][imask].center)
            # print(self.AllMaskProperties['V'][0].center[1])     
            

    #Auxialiar methods    
    @staticmethod
    def CalcAttPhase(att):
        att_linear = 10**(-att/10) #this is in linear
        att_phase = 2 * np.arccos(np.sqrt(att_linear)) # phase, from 0 to pi
        return(att_phase)    
        
    @staticmethod
    def binarycheckboard(width, height, spatial_frequency, scale=1, offset = False ,pol = 'HV'):
        y = np.arange(height, dtype = np.float64)#linspace(0,2*pi,1080)
        x = np.arange(width, dtype = np.float64)
        X,Y = np.meshgrid(x,y)
        
        GY = (((Y*spatial_frequency)%(1.0)>=.5)*scale).astype(np.float64)
        GX = (((X*spatial_frequency)%(1.0)>=.5)*scale).astype(np.float64)
        
        chBoard = abs(GX - GY)
        
        #if offset True, the pattern goes from -scale/2 to scale/2
        if offset == True:
            offset = scale/2
            chBoard -= offset
        
        if pol == 'H':
            chBoard[:,int(x.max()//2):-1] = 0
        elif pol == 'V':
            chBoard[:,0:int(x.max()//2)] = 0

        return(chBoard)
    
    # these two functions have hopefully been replace by the convert_phase_to_int8() fucntion and the     
    @staticmethod
    def getAngle(cmplxarray):
        im = cmplxarray.imag
        re = cmplxarray.real
        pi=np.pi
        # cc = ne.evaluate('arctan2(im,re)% (2 * pi)')
        cc = ne.evaluate('arctan2(im,re)')
        return(cc)
    #25 ms for this piece of code.
    @staticmethod
    def phaseTolevel(phasemask, aperture = 1): #phase mask to a level mask ready to send to the LCOS
        pi = np.pi
        # operate = ne.evaluate('mod(128 * phasemask / pi, 256)')
        # operate = np.mod(128 * phasemask / pi, 256)
        # phase_idx = int(phase_levels * (arg + np.pi) / (2 * np.pi)) % phase_levels # from modelab
        # operate = ne.evaluate(('(255*((phasemask+pi)/(2*pi)))%255')) # This is for -pi to pi

        operate = ne.evaluate('(phasemask * 128 / pi) % 256')
        # operate = ne.evaluate('255*((phasemask+pi)/(2*pi)))') # This is for -pi to pi
        # operate = ne.evaluate('255*((phasemask)/(2*pi))')# this is for 0 to 2pi
        # operate = ne.evaluate('((pi - phasemask) / (2 * pi)) * 255') # This is for -pi to pi


        operate_uint8 = operate.astype(np.uint8) # Lost of information since it will round to the smallest - as well it will wrap values ourside -pi and pi (This happen when HV mask superimpose and no phase wrap has been done)
        error = ne.evaluate('operate-operate_uint8')
        operate_uint8[error>0.5]+=1
        # level_array = np.round((phasemask % (2 * np.pi)) * (256 / (2 * np.pi))).astype(np.uint8) #operate_uint8 * aperture
        level_array = operate_uint8 * aperture
        
        return(level_array)
        
    
    #This is slow, no thought to recalculate apertures all the time
    @staticmethod
    def aperture(diameter,center, LCOS_size, px_size):
        pxY = LCOS_size[0]
        pxX = LCOS_size[1]
        x = np.arange(pxX) - center[0] 
        y = np.arange(pxY) - center[1] 
        X,Y = np.meshgrid(x,y)
        r = (np.sqrt(X**2 + Y**2))
        m = np.copy(X) * 0
        radius_px = np.floor(diameter/(2*px_size)) 
    
        m[r<=(radius_px)] = 1
        return(m)
    
    def CourseSweepAcrossSLM(self,channel,flipCount):
        self.LCOS_Clean()
        # flipMin=//2-flipCount//2
        flipMin=0
        flipMax=self.slmHeigth//2+flipCount//2
        flipMax=self.slmWidth//2+flipCount//2

        #Left to right sweep
        for iflip in range(0,self.slmWidth,flipCount):
            PiFlip_cmplx =np.ones((self.slmHeigth,self.slmWidth),dtype=complex)*np.exp(0.0*1j*np.pi)
       
            PiFlip_cmplx[:,0:flipMin+iflip]=PiFlip_cmplx[:,0:flipMin+iflip]*np.exp(1j*np.pi)
    
            ArryForSLM=self.phaseTolevel(np.angle(PiFlip_cmplx))
            self.Write_To_Display(ArryForSLM, channel)

           
            
        # top to bottom sweep    
        for iflip in range(0,self.slmHeigth,flipCount):
            PiFlip_cmplx =np.ones((self.slmHeigth,self.slmWidth),dtype=complex)*np.exp(0.0*1j*np.pi)
       
            PiFlip_cmplx[0:flipMin+iflip,:]=PiFlip_cmplx[0:flipMin+iflip,:]*np.exp(1j*np.pi)
    
            ArryForSLM=self.phaseTolevel(np.angle(PiFlip_cmplx))
            self.Write_To_Display(ArryForSLM, channel)

        
        self.LCOS_Clean(channel)
        return
    
def convert_phase_to_uint8(arr):
    """
    Convert the phase of each non-zero complex element in a 2D array 
    into a uint8 value in the range 0 to 255.
    
    For each element where the complex value is not 0+0j, the function:
      1. Computes the phase using numexpr's evaluated arctan2 (in radians)
      2. Converts the phase from the [-pi, pi] range to the [0, 255] range
         using the formula:
             scaled_value = round(((angle + pi) / (2*pi)) * 255)
    Elements that are 0+0j are left as 0.
    
    Parameters:
        arr (np.ndarray): A 2D NumPy array of complex numbers.
    
    Returns:
        np.ndarray: A 2D uint8 array with the converted phase values.
    """
    # Preallocate output array, defaulting all values to 0.
    out = np.zeros(arr.shape, dtype=np.uint8)
    
    # Create a mask for entries that are not exactly 0+0j.
    mask = (arr != 0)
    
    if np.any(mask):
        # Extract the real and imaginary parts for the masked entries.
        re = arr.real[mask]
        im = arr.imag[mask]
        
        # Compute the phase angle using numexpr.
        # Note: arctan2 returns values in [-pi, pi].
        angle = ne.evaluate("arctan2(im, re)", local_dict={"im": im, "re": re})
        
        # Convert angle to the range [0, 255]:
        #   scaled_value = ((angle + pi) / (2*pi)) * 255
        # We use np.rint to round to the nearest integer.
        out[mask] = np.rint((angle + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    
    return out
# if __name__ == '__main__':
    
#     import sys
#     from PyQt5.QtWidgets import *
    
#     from pylab import *
    
#     app = QApplication(sys.argv)
    
#     LCOSobject = LCOS(screen=1, mask_size=(960,960))
    
#     app.exec()