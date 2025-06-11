import numpy as np
import matplotlib.pyplot as plt

def LensPhaseProf(focalLen,wavelength,XGrid,YGrid):
    focusFactorX=(np.pi/(wavelength*focalLen))
    focusFactorY=(np.pi/(wavelength*focalLen))
    LensProf=np.exp(1j*(( (focusFactorX)*XGrid**2) + (focusFactorY)*YGrid**2))
    return LensProf
def TiltPhaseProf(tiltXdeg,tiltYdeg,wavelength,XGrid,YGrid):
    #This function takes in degrees since it is easier to understand
    k0 = 2.0*np.pi/wavelength;
    pixelSize= XGrid[0,1]-XGrid[0,0]
    #k_i_limit=2*pixelSize/(np.pi*wavelength)
    #print(k_i_limit)
    #This is the limit that the angle can be
    theta_limit= np.arcsin(wavelength/(2.0*pixelSize))*180.0/np.pi
    print('Angle limit for pixel ',theta_limit)
    
    ky0 = k0*np.sin(tiltYdeg* np.pi/180)
    kx0 = k0*np.sin(tiltXdeg* np.pi/180) 
    tiltProf=np.exp(1j*(( (kx0)*XGrid) + (ky0)*YGrid))
    return tiltProf

def PiFlipMasks(Nx,Ny,planeCount,PlotMasks):
    if planeCount>1:
        modeCountTotal= (planeCount-1)*2
        Masks=np.ones([modeCountTotal,planeCount,Ny,Nx],dtype=np.csingle)
        PiFlipHorzt=np.ones([Ny,Nx],dtype=np.csingle)
        PiFlipVert=np.ones([Ny,Nx],dtype=np.csingle)
        # print(np.shape(PiFlipHorzt))
        PiFlipHorzt[:,0:int((Nx/2))]=PiFlipHorzt[:,0:int((Nx/2))]*np.exp(1j*np.pi/2)
        PiFlipHorzt[:,int((Nx/2)):Nx]=PiFlipHorzt[:,int((Nx/2)):Nx]*np.exp(1j*(np.pi/2+np.pi))
        PiFlipVert[0:int((Nx/2)),:]=PiFlipVert[0:int((Nx/2)),:]*np.exp(1j*np.pi/2)
        PiFlipVert[int((Ny/2)):Ny,:]=PiFlipVert[int((Ny/2)):Ny,:]*np.exp(1j*(np.pi/2+np.pi))

        imodeIdx=0
        for iplane in range(planeCount-1):
            for imode in range(2):
                if (imode==0):
                    Masks[imodeIdx,planeCount-1,:,:]=PiFlipHorzt
                    Masks[imodeIdx,iplane,:,:]=PiFlipVert
                else:
                    Masks[imodeIdx,planeCount-1,:,:]= PiFlipVert
                    Masks[imodeIdx,iplane,:,:] = PiFlipHorzt
                imodeIdx=imodeIdx+1
        if (PlotMasks):
            for imode in range(modeCountTotal):
                plt.figure(imode)
                for iplane in range(planeCount):
                    plt.subplot(1,8,iplane+1)
                    plt.imshow(np.angle(Masks[imode,iplane,:,:]))
                    plt.axis('off')  
    else:
        modeCountTotal=2
        Masks=np.ones((modeCountTotal,planeCount,Ny,Nx),dtype=np.csingle)
        top_phase = np.pi
        bottom_phase = 2*np.pi
        # Create the mask top half
        Masks[0, 0, :Ny//2, :] =Masks[0, 0, :Ny//2, :] *np.exp(1j * top_phase)
       # # Create the mask bottom half
        Masks[0, 0, Ny//2:, :] = Masks[0, 0, Ny//2:, :] *np.exp(1j * bottom_phase)
        
        # Create the mask Right half
        right_phase = np.pi
        left_phase = 2*np.pi
        Masks[1, 0, :, Nx//2:] = Masks[1, 0, :, Nx//2:]*np.exp(1j * right_phase)
        # # Create the mask left half
        Masks[1, 0, :, :Nx//2] =Masks[1, 0, :, :Nx//2]*np.exp(1j * left_phase)
        if (PlotMasks):
            for imode in range(modeCountTotal):
                plt.figure(imode)
                plt.imshow(np.angle(Masks[imode,iplane,:,:]))
                plt.axis('off')  

    return Masks

def TiltMask(tiltXdeg,tiltYdeg,wavelength,Nx,Ny,pixelSize,PlotMasks):
    ymin = (((-(Ny - 1)) / 2.0)) * pixelSize;
    ymax = (((Ny - 1) / 2.0)) * pixelSize;
    y=np.linspace(ymin,ymax,Ny)
    xmin = (((-(Nx - 1)) / 2.0)) * pixelSize;
    xmax = (((Nx - 1) / 2.0)) * pixelSize;
    x=np.linspace(xmin,xmax,Nx)
    XGrid, YGrid= np.meshgrid(x,y)
    TiltPhase=TiltPhaseProf(tiltXdeg,tiltYdeg,wavelength,XGrid,YGrid)
    norm = (np.sqrt(sum(sum(np.abs(TiltPhase)**2))*pixelSize*pixelSize))
    TiltPhase=TiltPhase/norm
    # print(np.sqrt(sum(sum(np.abs(FocalPhase)**2))*pixelSize*pixelSize))
    if (PlotMasks):
        plt.figure()
        plt.imshow((np.angle(TiltPhase)))
        plt.figure()
        plt.imshow((abs(TiltPhase)))
    Masks=np.ones((1,1,Ny,Nx),dtype=np.csingle)
    Masks[0,0,:,:]=TiltPhase
    return Masks 

def LenMask(focalLen,wavelength,Nx,Ny,pixelSize,PlotMasks):
    ymin = (((-(Ny - 1)) / 2.0)) * pixelSize;
    ymax = (((Ny - 1) / 2.0)) * pixelSize;
    y=np.linspace(ymin,ymax,Ny)
    xmin = (((-(Nx - 1)) / 2.0)) * pixelSize;
    xmax = (((Nx - 1) / 2.0)) * pixelSize;
    x=np.linspace(xmin,xmax,Nx)
    XGrid, YGrid= np.meshgrid(x,y)
    FocalPhase=LensPhaseProf(focalLen,wavelength,XGrid,YGrid)
    norm = (np.sqrt(sum(sum(np.abs(FocalPhase)**2))*pixelSize*pixelSize))
    FocalPhase=FocalPhase/norm
    # print(np.sqrt(sum(sum(np.abs(FocalPhase)**2))*pixelSize*pixelSize))
    if (PlotMasks):
        plt.figure()
        plt.imshow((np.angle(FocalPhase)))
        plt.figure()
        plt.imshow((abs(FocalPhase)))
    Masks=np.ones((1,1,Ny,Nx),dtype=np.csingle)
    Masks[0,0,:,:]=FocalPhase
    return Masks 

def SpiralMask(SpiralNum,Nx,Ny,pixelSize,PlotMasks):
    # Pixel counts of the masks and simulation in 
    # Setup mask Cartesian co-ordinates/0.5 pixel 
    ymin = (((-(Ny - 1)) / 2.0)) * pixelSize;
    ymax = (((Ny - 1) / 2.0)) * pixelSize;
    y=np.linspace(ymin,ymax,Ny)
    xmin = (((-(Nx - 1)) / 2.0)) * pixelSize;
    xmax = (((Nx - 1) / 2.0)) * pixelSize;
    x=np.linspace(xmin,xmax,Nx)
    XGrid, YGrid= np.meshgrid(x,y)
    THGrid=np.arctan2(YGrid,XGrid)

    l=SpiralNum
    spiralPhasePaten = np.empty(np.shape(XGrid), dtype=complex)
    THGrid=np.arctan2(YGrid,XGrid)
    # spiralPhasePaten=np.exp(complex(0.0,1.0) *(  l * THGrid + (l + 2 * m + 1) ));
    spiralPhasePaten=np.exp(complex(0.0,1.0) *(  l * THGrid  ));
    if (PlotMasks):
        plt.figure()
        plt.imshow((np.angle(spiralPhasePaten)))

    Masks=np.ones((1,1,Ny,Nx),dtype=np.csingle)
    Masks[0,0,:,:]=spiralPhasePaten
    return Masks

    