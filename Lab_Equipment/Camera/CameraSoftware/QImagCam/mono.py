# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:40:44 2023

@author: cfai2304
"""
import serial
from calibration import to_wavelength
DEFAULT_BAUDRATE = 9600
DEFAULT_PORT = "COM11"
DEFAULT_TIMEOUT = 10 # ms

class Mono:
    """Control class for a Princeton instruments HRS300 monochromator"""
    
    def __init__(self, port=DEFAULT_PORT):
        self.ser=serial.Serial(port,baudrate=DEFAULT_BAUDRATE,
                               parity=serial.PARITY_NONE,
                               bytesize=serial.EIGHTBITS,
                               stopbits=serial.STOPBITS_ONE,
                               timeout=DEFAULT_TIMEOUT)
        self.clear_ser()
        
    def close(self):
        """Close monochromator connection"""
        self.ser.close()
        
    def clear_ser(self):
        """Clear port buffer"""
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
    def wait_for_serial(self):
        """ Block until 'ok' received from mono"""
        flagStop = 0
        response = ""
        response = self.ser.read(1).decode()
        while flagStop == 0:
            response += self.ser.read(1).decode()
            if "ok" in response:
                flagStop = 1
        self.clear_ser()
        return response
    
    def get_gratings(self):
        """Get list of installed gratings"""
        self.ser.write("?GRATINGS".encode()+ b"\x0d")
        return self.wait_for_serial()
    
    def set_grating(self, num):
        """Set grating in num'th position as active"""
        self.ser.write("{:0d} GRATING".format(num).encode() + b"\x0d")
        return self.wait_for_serial()
    
    def set_wavelength(self, w):
        """Move mono to wavelength w, in nm"""
        self.ser.write("{:.3f} GOTO".format(w).encode()+ b"\x0d")
        return self.wait_for_serial()
    
    def set_raman_shift(self, r, w0=532):
        """Move mono to raman shift r, relative to laser wavelength w0"""
        w = to_wavelength(r, w0)
        return self.set_wavelength(w)

if __name__ == "__main__":
    m = Mono()
    print(m.set_grating(3))
    m.close()
    
    