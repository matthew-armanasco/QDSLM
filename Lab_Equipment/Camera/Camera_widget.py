from Lab_Equipment.Config import config 
import Lab_Equipment.Camera.CameraObject as CamForm

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

# Make sure that CamForm is imported if not already
import Lab_Equipment.Camera.CameraObject as CamForm
def create_camera_widget(CamObj: CamForm.GeneralCameraObject):
    # Create widgets
    widget_Exposure = widgets.FloatText(
        value=CamObj.Exposure,
        description='Exposure', 
        layout=widgets.Layout(width='160px')
    )

    # Create button widget
    GetFrame_button = widgets.Button(description="Get Frame")
    CaptureMode_button = widgets.Button(description="Capture Mode: Continuous/Single ",layout=widgets.Layout(width='250px'))
   

   
    fig, ax = plt.subplots()
    fig.canvas.header_visible = False
    frame = CamObj.GetFrame()
    image_display = ax.imshow(frame)
    plt.axis("off")  # Hide axes
    # Function to update the plot when button is clicked
    def update_frame(_):
        frame = CamObj.GetFrame()
        image_display.set_data(frame)
        fig.canvas.draw_idle()  # Redraw figure without clearing widgets

    # Function to update exposure
    def on_Exposure_change(change):
        CamObj.SetExposure(change['new'])
        update_frame(None)
    def on_CaptrueMode_change(change):
        if CamObj.ContinuesMode.is_set():
            CamObj.SetSingleFrameCapMode()
        else:   
            CamObj.SetContinousFrameCapMode()
        
    
    

    # Attach the observer to widget_Exposure and button
    widget_Exposure.observe(on_Exposure_change, names='value')
    GetFrame_button.on_click(update_frame)
    CaptureMode_button.on_click(on_CaptrueMode_change)
    

    # Organize layout using a vertical box for controls
    grid = widgets.HBox([GetFrame_button,CaptureMode_button, widget_Exposure])

    return grid
