'''
Required packages
numpy pip install numpy
matplotlib pip install matplotlib
scipy pip install scipy
opencv pip install opencv-python
ipywidgets pip install ipywidgets
numexpr pip install numexpr
pip install PyQt5
pip install pyqt5-tools
pip install hdf5storage
serial control pip install pyserial
pyvisa pip install pyvisa
pyvisa backend pip install pyvisa-py
clr for dot net stuff pip install pythonnet
Thorlabs elliptec mount pip install thorlabs-elliptec
Thorlabs power meter need to be installed from the driver direcorty of the power meter go to directory and then run python setup.py install 
'''
import subprocess
import sys

# List of packages to install
packages = [
    "numpy",
    "matplotlib",
    "scipy",
    "opencv-python",
    "ipywidgets",
    "numexpr",
    "hdf5storage",
    "pyserial",
    "pyvisa",
    "pyvisa-py",
    "pythonnet",
    "thorlabs-elliptec",
    "screeninfo",
    "numba",
    "ipykernel",
    "ipympl",
    "ThorlabsPM100",
    "cma"
]

def install_packages(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    install_packages(packages)