import NKTP_DLL
import time

# Open the COM port
# Not nessesary, but would speed up the communication, since the functions does
# not have to open and close the port on each call
laserCOM = 'COM6'
openResult = NKTP_DLL.openPorts(laserCOM, 0, 0)
print('Opening the comport:', NKTP_DLL.PortResultTypes(openResult))


# Example - Turn on emission on FS50 by setting register 0x30 = 4
# See SDK Instruction Manual section 6.20
wrResult = NKTP_DLL.registerWriteU8(laserCOM, 15, 0x30, 4, -1) 
print('Turn on emission:', NKTP_DLL.RegisterResultTypes(wrResult))


# Example - Power to 100 % by setting register 0x99 = 100 / multiplier
wrResult = NKTP_DLL.registerWriteU16(laserCOM, 15, 0x99, int(100/0.1), -1) 
print('Change power level:', NKTP_DLL.RegisterResultTypes(wrResult))

# Wait 10 seconds for the laser to power up & settle
time.sleep(10)

# Example - Initiate auto-tune by setting register 0xE7 = 1
wrResult, value = NKTP_DLL.registerWriteReadU8(laserCOM, 15, 0xE7, 1, -1)
print('Auto-tune command sent:', NKTP_DLL.RegisterResultTypes(wrResult))

# Simple error checking - register will return values if auto-tune disabled
# See SDK Instruction Manual section 6.20
if value == 4:
    print('Error 4: Laser emission off')
elif value == 5:
    print('Error 5: Low peak signal')
else:
    print('Auto-tune in progress', end = '')
    
    # While loop continually checks register is in running state
    while value == 1:
        print('.', end = '')
        time.sleep(10)
        
        # Example - Read auto-tune status from register 0xE7
        rdResult, value = NKTP_DLL.registerReadU8(laserCOM, 15, 0xE7, -1)
    
    print('')
    # Example - Read dispersion values from compound register 0xB6
    # See SDK Instruction Manual section 6.20
    rdResult, D2 = NKTP_DLL.registerReadF32(laserCOM, 15, 0xB6, 0)
    rdResult, D3 = NKTP_DLL.registerReadF32(laserCOM, 15, 0xB6, 4)
    rdResult, D4 = NKTP_DLL.registerReadF32(laserCOM, 15, 0xB6, 8)
    
    # Report of auto-tune result
    if value == 2:
        print('Auto-tune Success. New dispersion setpoints:')
    elif value == 3:
        print('Auto-tune timeout. Best dispersion setpoints:')
    elif value == 9:
        print('User terminated auto-tune. Last dispersion setpoints:')

    print('D2 = {}'.format(D2))
    print('D3 = {}'.format(D3))
    print('D4 = {}'.format(D4))
    
# Example - Power to 0 % by setting register 0x99 = 0
wrResult = NKTP_DLL.registerWriteU16(laserCOM, 15, 0x99, 0, -1) 
print('Change power level:', NKTP_DLL.RegisterResultTypes(wrResult))

# Example - Turn off emission on FS50 by setting register 0x30 = 0
wrResult = NKTP_DLL.registerWriteU8(laserCOM, 15, 0x30, 0, -1) 
print('Turn off emission:', NKTP_DLL.RegisterResultTypes(wrResult))

# Close the COMport
closeResult = NKTP_DLL.closePorts(laserCOM)
print('Close the comport:', NKTP_DLL.PortResultTypes(closeResult))