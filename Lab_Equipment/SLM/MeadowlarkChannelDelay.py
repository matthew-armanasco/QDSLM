import serial

# Change the delay on SLM For colour red
# Delay   Command (hex)                                 Binary

# 0	      s 000c 04804d10                               0000 0100 1000 0000 0100 1101 0001 0000
# 4	      s 000c 24804d10 (7.217V, 30.322C) Seems good  0010 0100 1000 0000 0100 1101 0001 0000
# 6	      s 000c 34804d10 (Default?)                    0011 0100 1000 0000 0100 1101 0001 0000
# 8	      s 000c 44804d10                               0100 0100 1000 0000 0100 1101 0001 0000
# 10	    s 000c 54804d10                               0101 0100 1000 0000 0100 1101 0001 0000
# 12	    s 000c 64804d10                               0110 0100 1000 0000 0100 1101 0001 0000
# 14	    s 000c 74804d10                               0111 0100 1000 0000 0100 1101 0001 0000
# (Pushing it this way delay seems to make Grey 50 less noisy, 7.158V)


# Change the delay on SLM For colour green
# Delay   Command (hex)                                 Binary

# 0	      s 000c 0480 4D50                               0000 0100 1000 0000 0100 1101 0101 0000
# 4	      s 000c 2480 4D50 (7.217V, 30.322C) Seems good  0010 0100 1000 0000 0100 1101 0101 0000
# 6	      s 000c 34804d50 (Default?)                    0011 0100 1000 0000 0100 1101 0101 0000
# 8	      s 000c 44804d50                               0100 0100 1000 0000 0100 1101 0101 0000
# 10	    s 000c 54804d50                               0101 0100 1000 0000 0100 1101 0101 0000
# 12	    s 000c 64804d50                               0110 0100 1000 0000 0100 1101 0101 0000
# 14	    s 000c 74804d50                               0111 0100 1000 0000 0100 1101 0101 0000
# (Pushing it this way delay seems to make Grey 50 less noisy, 7.158V)


def SetColourChannelAndDelay(Comport,ColourChannel,Delay):

    ser = serial.Serial(Comport, 115200) # open the serial port with the given port name and baud rate
    
    if Delay==0:
        delaystr='0'
    elif Delay==4:
        delaystr='2'
    elif Delay==6:
        delaystr='3'
    elif Delay==8:
        delaystr='4'
    elif Delay==10:
        delaystr='5'
    elif Delay==12:
        delaystr='6'
    elif Delay==14:
        delaystr='7'
    else:
        print("Invaild delay. must be 0,4,6,8,10,12 or 14 with that exact spelling")
        return

    if ColourChannel=="Red":
        ColourChannelstr="d10"
    elif ColourChannel=="Green":
        ColourChannelstr="d50"
    elif ColourChannel=="Blue":
        ColourChannelstr="d70"
    else: 
        print("Invaild colour. must be Red Green or Blue with that exact spelling")
        return
    seriallin = "s000c"+ delaystr +"4804"+ColourChannelstr
    ser.write(seriallin.encode()) # write the command as a byte string

    response = ser.readline().decode() # Read the response from the serial port
    print(response) 
    ser.close() # close the serial port