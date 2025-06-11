import numpy as np
from math import factorial
# import hdf5storage
import os
import Lab_Equipment.Config.config as config
import matplotlib.pyplot as plt
import numexpr as ne
from enum import IntEnum


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

class ZernCoefs(IntEnum):
    PISTON = 0
    TILTX = 1
    TILTY = 2
    ASTIGX = 3
    DEFOCUS = 4
    ASTIGY = 5
    TREFOILX = 6
    COMAX = 7
    COMAY = 8
    TREFOILY = 9
    SPHERICAL = 12
# zern_coefs
# 0=piston 0 0
# 1=Tiltx -1 1
# 2=Tilty 1 1
# 3=Astigx -2 2
# 4=Defocus 0 2
# 5=Astigy 2 2
# 6=Trefilx -3 3
# 7=Comax -1 3
# 8=Comay  1 3
# 9=Trefoily 3 3 
# 12=Spherical 0 4
class Zernikes:

    def __init__(self, max_zernike_radial_number = 4, Nx = 1024, Ny = 1024, aperture_radius_in_m = None,
                 pixelSize = 9.2e-6,wavelength=1565e-9, load_modelab = False):
        self.iplane=0
        self.b_iplane=self.iplane
        self.max_zernike_radial_number = max_zernike_radial_number
        self.wavelength = wavelength
        # mask midpoint
        half_width = Nx//2.0
        half_height = Ny//2.0
        

        # Makes a polar grid with r = 1 at r_norm (in pixels). This properly fits the Zernike to the required aperture
        # x,y = np.meshgrid(np.linspace(0, Nx - 1, Nx, dtype = np.float32),\
        #                 np.linspace(0, Ny - 1, Ny, dtype = np.float32))
        # rho, phi = cart2pol(x - half_width, y - half_height)
        x = np.linspace(-1, 1, Nx)
        y = np.linspace(-1, 1, Ny)
        X, Y = np.meshgrid(x, y)
        rho, phi = cart2pol(X, Y)
        rho = np.clip(rho, 0, 1)

        self.Dims = [Nx,Ny] # Defines the array size on which we calculate the Zernikes (same as mask_size_x_in_pixels & mask_size_y_in_pixels if square)
        self.Nx=Nx
        self.Ny=Ny
        if aperture_radius_in_m is None:  
            # self.aperture_radius_in_m=np.sqrt((pixelSize*(self.Nx//2))**2+(pixelSize*(self.Ny//2))**2)
            self.aperture_radius_in_m=pixelSize*(self.Nx//2)
            # print(r_norm)
        else:
            self.aperture_radius_in_m = aperture_radius_in_m
            
        # r_norm = self.aperture_radius_in_m
        # rho = rho/r_norm  
        # rho =rho*SLM_pixel_size_in_um
        self.zernCount,self.zernike_basis_list, self.zernike_basis_array = self.generate_zernike_basis(rho, phi, max_zernike_radial_number)
        self.zern_coefs=np.zeros(self.zernCount)
        if load_modelab == True:
            # self.load_modelab_coefs()
            print("modelab is dead to me may it burn in hell")
  
        self.make_zernike_fields()


    def zernikePoly_nm(self, rho, phi, n, m):
        #""" Calculate the radial component of Zernike polynomial (n, m) """
        Z_Poly = np.zeros_like(rho)
        for k in range((n - abs(m)) // 2 + 1):
            coef = (-1)**k * np.math.factorial(n - k)
            coef /= np.math.factorial(k) * np.math.factorial((n + abs(m)) // 2 - k) * np.math.factorial((n - abs(m)) // 2 - k)
            Z_Poly += coef * rho**(n - 2*k)
        #""" Calculate Zernike polynomial (n, m) on a grid of polar coordinates rho and phi """
        if m > 0:
            Z_Poly *= np.cos(m * phi)
        elif m < 0:
            Z_Poly *= np.sin(-m * phi)
        return Z_Poly
    

    def generate_zernike_basis(self, rho, phi, max_radial_idx):
        zernike_count = 0
        zernike_basis_list = [] # list of [zernike_profile, n, m]
        
        for n in range(max_radial_idx + 1):
            mm = range(-n,n+1,2)
            for m in mm:
                zernike_basis_list.append([self.zernikePoly_nm(rho, phi, n, m,), n, m])
                zernike_count += 1
        # each slice of the 3D array represents a particular Zernike profile. I create this to avoid using zernikes list which is useful but slow in calculations 
        zernike_basis_array = np.zeros((zernike_count, rho.shape[0],rho.shape[1]), dtype = np.float32) 
        for idx in range(zernike_count):
            zernike_basis_array[idx,:,:] = zernike_basis_list[idx][0] # [0] is the index of the zernike profile array
    
        return (zernike_count,zernike_basis_list, zernike_basis_array)
    
    def make_zernike_fields(self): 
        # set the total Zernike phase array in H,V pol to 0
        zern_phase = np.zeros((self.Ny,self.Nx), dtype = np.float32)
        # calculate the phase coming from Zernikes and modelab coefficients (piston, tilt_x and tilt_y are not part of this, that's why there is  (-3) in the loop)
        for izern in range(self.zernCount):
            if izern == ZernCoefs.TILTX or izern == ZernCoefs.TILTY:
                zernikeWieght = (2*np.pi*self.aperture_radius_in_m/self.wavelength)*np.deg2rad(self.zern_coefs[izern] )
            elif izern == ZernCoefs.DEFOCUS:
                if self.zern_coefs[izern] != 0:
                    zernikeWieght = (np.pi*self.aperture_radius_in_m**2)/(self.wavelength*(self.zern_coefs[izern]))
                else:
                    zernikeWieght = 0
            else:
                zernikeWieght = self.zern_coefs[izern] 
            zern_phase += ne.evaluate('w_i*zern_i', {'w_i': zernikeWieght, 'zern_i': self.zernike_basis_array[izern,:,:]})
        # Make the fields from zern_phase
        self.field = ne.evaluate('exp(1j*pi*zern_phase_h)', {'zern_phase_h': zern_phase, 'pi': np.pi}).astype(np.complex64)
        return self.field




        
#This is a good function but it needs to be changed so the that masks setting file not a modelab compatable as modelab is dead to me.
# I am not going to use it until it is properly implemented
#     def load_modelab_coefs(self):
        
#         # filename = os.path.join(python_settings.mask_settings_path,'MASK_SETTINGS.mat')
#         filename = os.path.join(config.mask_settings_path,'MASK_SETTINGS.mat')

#         # filename = os.path.join(r'/Users/s4356803/Documents/PhD/Codes/PythonCode/SLMPythonMODULES','MASK_SETTINGS.mat')
        
#         f = hdf5storage.loadmat(filename)

#         # ================ READ MASK CENTERS ====================================
#         #Masks centers are no longer kept in zernike as it is not really associated with zernike polynomials
#         # masks_centers_x = f['ZERNIKE_CX']
#         # masks_centers_y = f['ZERNIKE_CY']

#         # self.h_mask_center_x = int(masks_centers_x[0][0][0])
#         # self.h_mask_center_y = int(masks_centers_y[0][0][0])

#         # self.v_mask_center_x = int(masks_centers_x[0][0][1])
#         # self.v_mask_center_y = int(masks_centers_y[0][0][1])

#         zern_coefs = f['ZERNIKE_COEF'].astype(np.float32) # convert from string to float
        
#         zern_h = zern_coefs[0][0][0]
#         zern_v = zern_coefs[0][0][1]

#         # ================ READ PISTON and TILTS ====================================
#         self.piston_h = zern_h[0]
#         self.tilt_x_h = zern_h[1]
#         self.tilt_y_h = zern_h[2]

#         self.piston_v = zern_v[0]
#         self.tilt_x_v = zern_v[1]
#         self.tilt_y_v = zern_v[2]

#         # =========== READ THE REST OF ZERNIKES ==========================
#         self.n = []
#         self.m = []
#         self.w_h = zern_h[3:] #we start at index 3 because the piston, tiltx and tilt y is not included 
#         self.w_v = zern_v[3:]

#         for n_idx in range(2, self.max_zernike_radial_number + 1):
#                 mm = range(-n_idx,n_idx + 1,2)
#                 for m_idx in mm: 
#                     self.n.append(n_idx)
#                     self.m.append(m_idx)

#         self.zern_idxs = np.asarray([(n*(n+2) + m)//2 for n,m in zip(self.n,self.m)]) # This does not contain piston and tilt

 



# if __name__ == '__main__':
#     print('Module testing started!')
#     z = Zernikes(max_zernike_radial_number = 9)

    # import time
    # from multiprocessing import Pool

    # p = Pool(processes = 16)

    # data = range(2000000)

    # tic = time.time()
    # result = p.map(warmup, data)
    # toc = time.time()
    # print(f'The warm-up run {toc - tic}')
    # # print(result)

    # tic = time.time()  
    # p.map(z, data)
    # toc = time.time()
    # print(f'The benchmark run {toc - tic}')

    # tic = time.time()
    # for x in range(2000000):
    #     x*x
    # toc = time.time()
    # print(f'The single-threaded run {toc - tic}')    
    
    
