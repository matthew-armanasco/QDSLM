import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
import cv2
import numpy as np
import Lab_Equipment.SLM.pyLCOS as pyLCOS

def create_slm_widget(slm:pyLCOS.LCOS, pol="V", imask=0, channel="Red"):
    # Create widgets
    #####################################
    # Drop down  boxes
    ####################################
    widget_channel = widgets.Dropdown(
        options=[('Red SLM', "Red"), ('Green SLM', "Green"),('Blue SLM', "Blue")],
        value=channel, description='Channel',
        layout=widgets.Layout(width='200px')
    )
    widget_pol = widgets.Dropdown(
        options=[('H', "H"), ('V', "V")],
        value=pol, description='pol',
        layout=widgets.Layout(width='140px')
    )
    #####################################
    # Check boxes
    ####################################
    widget_PolEnableChecBox = widgets.Checkbox(
    value=True,
    description='Enable Pol',
    disabled=False,indent=False,
    layout=widgets.Layout(width='150px')
    
    
)
    widget_MaskEnableChecBox = widgets.Checkbox(
    value=True,
    description='Enable Mask',
    disabled=False,indent=False,
    layout=widgets.Layout(width='150px'))

    #####################################
    # Value boxes
    ####################################
    widget_Plane = widgets.IntText(value=0, 
        description='Plane', 
        layout=widgets.Layout(width='140px'))
    widget_Mode_H = widgets.IntText(value=0, 
        description='Mode_H', 
        layout=widgets.Layout(width='140px'))
    widget_Mode_V = widgets.IntText(value=0, 
        description='Mode_V', 
        layout=widgets.Layout(width='140px'))
    
    widget_XCenter = widgets.IntText(
        value=slm.AllMaskProperties[channel][pol][imask].center[1],
        description='X Center', layout=widgets.Layout(width='160px')
    )
    widget_YCenter = widgets.IntText(
        value=slm.AllMaskProperties[channel][pol][imask].center[0],
        description='Y Center', layout=widgets.Layout(width='160px')
    )
    widget_XTilt = widgets.FloatText(
        value=slm.AllMaskProperties[channel][pol][imask].zernike.zern_coefs[1],
        step=0.001,
        description='X Tilt', layout=widgets.Layout(width='160px')
    )
    widget_YTilt = widgets.FloatText(
        value=slm.AllMaskProperties[channel][pol][imask].zernike.zern_coefs[2],
        step=0.001,
        description='Y Tilt', layout=widgets.Layout(width='160px')
    )
    widget_Piston = widgets.FloatText(
        value=slm.AllMaskProperties[channel][pol][imask].zernike.zern_coefs[0],
        step=2*np.pi/256,
        description='Piston', layout=widgets.Layout(width='160px')
    )
    widget_Defocus = widgets.FloatText(
        value=slm.AllMaskProperties[channel][pol][imask].zernike.zern_coefs[4],
        step=1,
        description='Defocus', layout=widgets.Layout(width='160px')
    )
    widget_RefreshTime = widgets.FloatText(
        value=slm.GLobProps[channel].RefreshTime*1e3,
        description='Refresh Rate (ms)', layout=widgets.Layout(width='180px')
    )
    
    widget_SweepStep = widgets.IntText(
        value=30,
        description='Sweep Step', layout=widgets.Layout(width='160px')
    )
    #######################################
    # text box
    ####################################
    widget_MaskFilename = widgets.Text(
        value=slm.MasksFilename,
        description="Mask File Name",
        layout=widgets.Layout(width='250px')
    )


    #####################################
    # Buttons
    ####################################
    update_button_currentSLM = widgets.Button(description='Update Current SLM', layout=widgets.Layout(width='150px'))
    update_button_AllSLM = widgets.Button(description='Update All SLM', layout=widgets.Layout(width='150px'))
    update_button_ClearSLM = widgets.Button(description='Clear SLM', layout=widgets.Layout(width='150px'))
    update_button_SetZernikeToZero = widgets.Button(description='Set All Zernike To Zero', layout=widgets.Layout(width='170px'))
    update_button_SetPlanesToEqualSpacing = widgets.Button(description='Set PlaneTo Equal Spacing', layout=widgets.Layout(width='170px'))
    update_button_ReversePlaneOrder = widgets.Button(description='Reverse Plane Order', layout=widgets.Layout(width='170px'))
    update_button_ViewDisplay = widgets.Button(description='View SLM Image', layout=widgets.Layout(width='170px'))
    Init_button_PiSweep = widgets.Button(description='Start Sweep', layout=widgets.Layout(width='170px'))
    
    Save_MaskProp_button = widgets.Button(description='Save Mask Props', layout=widgets.Layout(width='170px'))
    LoadPiFlipMasks_button =  widgets.Button(description='Load PI flip masks', layout=widgets.Layout(width='170px'))
    LoadMaskFile_button =  widgets.Button(description='Load mask files', layout=widgets.Layout(width='170px'))
   

    # Define event handlers (using closures to capture widget variables)
    def on_value_change(change):
        # Determine which widget changed and update accordingly.
        
        desc = change['owner'].description
        if desc == 'Mode_H':
            if widget_Mode_H.value > slm.polProps[widget_channel.value]['H'].modeCount - 1:
                widget_Mode_H.value = 0
            if widget_Mode_H.value < 0:
                widget_Mode_H.value = slm.polProps[widget_channel.value]['H'].modeCount - 1
        elif desc == 'Mode_V':
            if widget_Mode_V.value > slm.polProps[widget_channel.value]['V'].modeCount - 1:
                widget_Mode_V.value = 0
            if widget_Mode_V.value < 0:
                widget_Mode_V.value = slm.polProps[widget_channel.value]['V'].modeCount - 1
            # slm.setmask(widget_channel.value, widget_Mode.value)
        elif desc == 'X Center':
            slm.AllMaskProperties[widget_channel.value][widget_pol.value][widget_Plane.value].center[1] = change['new']
        elif desc == 'Y Center':
            slm.AllMaskProperties[widget_channel.value][widget_pol.value][widget_Plane.value].center[0] = change['new']
        elif desc == 'X Tilt':
            slm.AllMaskProperties[widget_channel.value][widget_pol.value][widget_Plane.value].zernike.zern_coefs[1] = change['new']
        elif desc == 'Y Tilt':
            slm.AllMaskProperties[widget_channel.value][widget_pol.value][widget_Plane.value].zernike.zern_coefs[2] = change['new']
        elif desc == 'Piston':
            slm.AllMaskProperties[widget_channel.value][widget_pol.value][widget_Plane.value].zernike.zern_coefs[0] = change['new']
        elif desc == 'Defocus':
            slm.AllMaskProperties[widget_channel.value][widget_pol.value][widget_Plane.value].zernike.zern_coefs[4] = change['new']
        elif desc == 'Refresh Rate (ms)':
           slm.GLobProps[widget_channel.value].RefreshTime = change['new']*1e-3
        
        slm.setmask(widget_channel.value,imode_H=widget_Mode_H.value,imode_V=widget_Mode_V.value)

    def on_button_click(event, update_all=False):
        if update_all:
            for ch in slm.ActiveRGBChannels:
                for ipol in ["H","V"]:

                    update_slm_properties(ch, ipol, widget_Plane.value)
        else:
            update_slm_properties(widget_channel.value, widget_pol.value, widget_Plane.value)

    def on_button_click_clearSLM(event):
        for ch in slm.ActiveRGBChannels:
            slm.LCOS_Clean(ch)

    def on_button_click_SetAllZernikeToZero(event):
        slm.ResetAllZernikesToZero(widget_channel.value)
        update_slm_properties(widget_channel.value, widget_pol.value, widget_Plane.value)
    
    def on_button_click_SetPlanesToEqualSpacing(event):
        slm.setCentersToEqualSpacing(widget_channel.value)
        update_slm_properties(widget_channel.value, widget_pol.value, widget_Plane.value)

    def on_button_click_ReversePlaneOrder(event):
        # slm.setCentersToEqualSpacing(widget_channel.value)
        print("Test")
        slm.mplc_reverse_order_mask_x_centers(widget_channel.value)
        update_slm_properties(widget_channel.value, widget_pol.value, widget_Plane.value)
        
    fig, ax = plt.subplots()
    fig.canvas.header_visible = False
    rgbimage=np.zeros((slm.slmHeigth, slm.slmWidth, 3), dtype=np.uint8)
    channelIdx=slm.GLobProps[widget_channel.value].rgbChannelIdx
    np.copyto(rgbimage[:,:,channelIdx],slm.FullScreenBuffer_int)
    rgb_image = cv2.cvtColor(rgbimage, cv2.COLOR_BGR2RGB)
    image_display = ax.imshow(rgb_image,aspect='auto')
    plt.axis("off")  # Hide axes
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # Function to update the plot when button is clicked
    def update_displayWidget(_):
        # rgb_image = cv2.cvtColor(slm.FullScreenBuffer_int, cv2.COLOR_BGR2RGB)
        channelIdx=slm.GLobProps[widget_channel.value].rgbChannelIdx
        np.copyto(rgbimage[:,:,channelIdx],slm.FullScreenBuffer_int)
        rgb_image = cv2.cvtColor(rgbimage, cv2.COLOR_BGR2RGB)
        image_display.set_data(rgb_image)
        fig.canvas.draw_idle()  # Redraw figure without clearing widgets

    # Observer callback for widget_Plane changes.
    def on_plane_change(change):
        # change['new'] is the new value of widget_Plane
        new_plane = change['new']
        # Sanity check for plane value
        
        if new_plane > slm.polProps[widget_channel.value][widget_pol.value].MaskCount - 1:
            widget_Plane.value = 0
        elif new_plane < 0:
            widget_Plane.value =  slm.polProps[widget_channel.value][widget_pol.value].MaskCount - 1
        # Now update the center widgets based on the new plane value.
        update_slm_properties(widget_channel.value, widget_pol.value, widget_Plane.value)

    def on_channel_change(change):  
         Channel = change['new']
         update_slm_properties(widget_channel.value, widget_pol.value, widget_Plane.value)
    
    def on_pol_change(change):
        pol = change['new']
        update_slm_properties( widget_channel.value, widget_pol.value, widget_Plane.value)

    def on_pol_Enable_change(change):
        if change['new']:
            if(widget_pol.value=="H"):# Turn the Vertical pol side of SLM off
                slm.polProps[widget_channel.value]['H'].polEnabled=True
            else:# Turn the Horizontial pol side of SLM off
                slm.polProps[widget_channel.value]['V'].polEnabled=True
        else:
            if(widget_pol.value=="H"):# Turn the Vertical pol side of SLM off
                slm.polProps[widget_channel.value]['H'].polEnabled=False
            else:# Turn the Horizontial pol side of SLM off
                slm.polProps[widget_channel.value]['V'].polEnabled=False

        update_slm_properties( widget_channel.value, widget_pol.value, widget_Plane.value)
    
    def on_Mask_Enable_change(change):
        slm.AllMaskProperties[widget_channel.value][widget_pol.value][widget_Plane.value].maskEnabled=widget_MaskEnableChecBox.value
        slm.setmask(widget_channel.value, imode_H=widget_Mode_H.value,imode_V=widget_Mode_V.value)


    def update_slm_properties(Channel, pol="V", imask=0):
        widget_XCenter.value = slm.AllMaskProperties[Channel][pol][imask].center[1]
        widget_YCenter.value = slm.AllMaskProperties[Channel][pol][imask].center[0]
        widget_Piston.value = slm.AllMaskProperties[Channel][pol][imask].zernike.zern_coefs[0]
        widget_XTilt.value = slm.AllMaskProperties[Channel][pol][imask].zernike.zern_coefs[1]
        widget_YTilt.value = slm.AllMaskProperties[Channel][pol][imask].zernike.zern_coefs[2]
        widget_Defocus.value = slm.AllMaskProperties[Channel][pol][imask].zernike.zern_coefs[4]
        if(pol=="H"):# Turn the Vertical pol side of SLM off
                widget_PolEnableChecBox.value=slm.polProps[Channel]['H'].polEnabled
        else:# Turn the Horizontial pol side of SLM off
                widget_PolEnableChecBox.value=slm.polProps[Channel]['V'].polEnabled
        widget_MaskEnableChecBox.value= slm.AllMaskProperties[Channel][pol][imask].maskEnabled
        slm.setmask(widget_channel.value, imode_H=widget_Mode_H.value,imode_V=widget_Mode_V.value)
    
    def InitialPiSweep(_):
        slm.CourseSweepAcrossSLM(widget_channel.value,widget_SweepStep.value)

    def SaveMaskProps(_):
        slm.saveMaskProperties(channel=widget_channel.value)

    def LoadPiFlipAlignmentMasks(_):
        slm.LoadPiFlipMasks(channel=widget_channel.value) 
    def LoadMaskFiles(_):
        slm.LoadMasksFromFile(Filename=widget_MaskFilename.value,channel=widget_channel.value,)


        
    # Attach the observer to widget_Plane.
    widget_Plane.observe(on_plane_change, names='value')
    widget_channel.observe(on_channel_change, names='value')
    widget_pol.observe(on_pol_change, names='value')
    widget_PolEnableChecBox.observe(on_pol_Enable_change, names='value')
    widget_MaskEnableChecBox.observe(on_Mask_Enable_change, names='value')
    
    
    

    # Register observers for the widgets
    for w in [widget_Mode_H,widget_Mode_V, widget_XCenter, widget_YCenter, widget_XTilt,
              widget_YTilt, widget_Piston, widget_Defocus,widget_RefreshTime]:
        w.observe(on_value_change, names='value')

    update_button_currentSLM.on_click(lambda event: on_button_click(event, update_all=False))
    update_button_AllSLM.on_click(lambda event: on_button_click(event, update_all=True))
    update_button_ClearSLM.on_click(on_button_click_clearSLM)
    update_button_SetZernikeToZero.on_click(on_button_click_SetAllZernikeToZero)
    update_button_SetPlanesToEqualSpacing.on_click(on_button_click_SetPlanesToEqualSpacing)
    update_button_ReversePlaneOrder.on_click(on_button_click_ReversePlaneOrder)
    update_button_ViewDisplay.on_click(update_displayWidget)
    Init_button_PiSweep.on_click(InitialPiSweep)
    Save_MaskProp_button.on_click(SaveMaskProps)
    LoadPiFlipMasks_button.on_click(LoadPiFlipAlignmentMasks)
    LoadMaskFile_button.on_click(LoadMaskFiles)
    
    # Organize the widgets using layout containers
    grid = widgets.GridBox(
        children=[
            widget_channel, widget_pol,widget_PolEnableChecBox,widget_Plane,widget_MaskEnableChecBox,
            widget_Mode_H, widget_Mode_V,
            widget_XCenter, widget_YCenter, widget_XTilt,
            widget_YTilt, widget_Piston, widget_Defocus,widget_RefreshTime,
            update_button_currentSLM, update_button_AllSLM, 
            LoadPiFlipMasks_button,
            update_button_ClearSLM,update_button_SetZernikeToZero,
            update_button_SetPlanesToEqualSpacing,update_button_ReversePlaneOrder,update_button_ViewDisplay,
            Save_MaskProp_button,
            widget_SweepStep,Init_button_PiSweep,
            widget_MaskFilename,LoadMaskFile_button],
         layout=widgets.Layout(
        grid_template_columns="repeat(5, 1fr)",
        grid_template_rows="repeat(5, auto)",
        grid_gap="10px"
    )
        # layout=widgets.Layout(
        #     grid_template_columns="repeat(5, 1fr)",
        #     grid_gap="10px"
        # )
    )
    return grid
