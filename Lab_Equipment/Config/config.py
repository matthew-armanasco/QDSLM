import sys
import os
import platform
WORKING_DIR = os.getcwd()
print("Current Directory:", WORKING_DIR)

LAB_EQUIPMENT_FOLDER="Lab_Equipment"


# More specific system check
system_name = platform.system()
if system_name == "Windows":
    print("This is a Windows system.")
    SLASH= "\\"
elif system_name == "Darwin":
    print("This is a macOS system.")
    SLASH= "/"
elif system_name == "Linux":
    print("This is a Linux system.")
    SLASH= "/"


EXPERIMENT_LIB_PATH = WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER
sys.path.append(EXPERIMENT_LIB_PATH)

SLM_LIB_PATH = WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"SLM"+SLASH
sys.path.append(SLM_LIB_PATH)

SLM_HOLO_LIB_PATH = WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"ZernikeModule"+SLASH
sys.path.append(SLM_HOLO_LIB_PATH)

CAMERA_LIB_PATH = WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"Camera"+SLASH
sys.path.append(CAMERA_LIB_PATH)

# CAMERA_LIB_PATH_ALLIEDCAM = WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH"Camera"\\AlliedCamera\\VmbPy\\"
CAMERA_LIB_PATH_ALLIEDCAM = CAMERA_LIB_PATH+SLASH+"AlliedCamera"+SLASH+"VmbPy"+SLASH
sys.path.append(CAMERA_LIB_PATH_ALLIEDCAM)

DIGHOLO_LIB_PATH = WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"digHolo"+SLASH
sys.path.append(DIGHOLO_LIB_PATH)

MPLC_LIB_PATH=  WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"AlignmentRoutines"+SLASH
sys.path.append(MPLC_LIB_PATH)

PATH_TO_MY_LIB_PATH=  WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"MyPythonLibs"+SLASH
sys.path.append(PATH_TO_MY_LIB_PATH)

MOKU_LIB_PATH=  WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"MokuLab"+SLASH
sys.path.append(MOKU_LIB_PATH)

PATH_TO_TIMETAGGER_SOFTWARE='C:\Program Files\Swabian Instruments\Time Tagger\driver\python'
if os.path.exists(PATH_TO_TIMETAGGER_SOFTWARE):
    sys.path.append(PATH_TO_TIMETAGGER_SOFTWARE)
else:
    print("Time Tagger Software not installed. If needed to go and get software from Swabin website.")
PATH_TO_TIMETAGGER_FOLDER=WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"TimeTagger"+SLASH
sys.path.append(PATH_TO_TIMETAGGER_FOLDER)

PATH_TO_THORLABS_KINESIS_SOFTWARE='C:\\Program Files\\Thorlabs\\Kinesis\\'
if os.path.exists(PATH_TO_THORLABS_KINESIS_SOFTWARE):
    sys.path.append(PATH_TO_THORLABS_KINESIS_SOFTWARE)
else:
    print("Kinesis Software not installed. If needed to go and get software from thorlabs website, DLL's are needed for the mount to work")

PATH_TO_THORLABS_FOLDER=WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"ThorlabsKCube"+SLASH
sys.path.append(PATH_TO_THORLABS_FOLDER)


PATH_TO_POWERMETER_FOLDER=WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"PowerMeter"+SLASH
sys.path.append(PATH_TO_POWERMETER_FOLDER)

PATH_TO_ELLTLMOUNT_FOLDER=WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"ElliptecThorlabMounts"+SLASH
sys.path.append(PATH_TO_ELLTLMOUNT_FOLDER)

PATH_TO_LASERS_FOLDER=WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"Laser"+SLASH
sys.path.append(PATH_TO_LASERS_FOLDER)

PATH_TO_OPTICSIMS_FOLDER=WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"OpticalSimulations"+SLASH
sys.path.append(PATH_TO_OPTICSIMS_FOLDER)

PATH_TO_DEFORMABLEMIRROR_SOFTWARE='C:\Program Files\Alpao\SDK\Samples\Python3\Lib64'
if os.path.exists(PATH_TO_DEFORMABLEMIRROR_SOFTWARE):
    sys.path.append(PATH_TO_DEFORMABLEMIRROR_SOFTWARE)
    print("The deformable mirror library only works with Python 3.7, please use a virtual environment with Python 3.7,\
        this will be fixed in the future")
else:
    print("Deformable Mirror Software not installed. If needed to go and get software from USB website.")
PATH_TO_DEFROMABLEMIRROR_FOLDER=WORKING_DIR+SLASH+LAB_EQUIPMENT_FOLDER+SLASH+"DeformableMirror"+SLASH
sys.path.append(PATH_TO_DEFROMABLEMIRROR_FOLDER)


# This might need to be uncommeted I am not sure yet need to test with camera
# TEST="C:\\Program Files\\Point Grey Research\\FlyCapture2\\bin64\\"
# sys.path.append(TEST)

