

from pyanc350.v2 import Positioner
import time

ax = {'x':0,'y':1,'z':2}
#define a dict of axes to make things simpler

anc = Positioner()
#instantiate positioner as anc
print('-------------------------------------------------------------')
# print('capacitances:')
# for axis in sorted(ax.keys()):
    # print(axis,anc.capMeasure(ax[axis]))
    
# print('moving to x = 2mm')
# anc.moveAbsolute(ax['x'],2000000)

print('frequency',anc.getFrequency(0))

print(anc.getDcLevel(0))

print(anc.getAmplitude(0))

# anc.frequency(0, 100)
# val = int(input('How many seconds u want to see axis for: '))

val2 = int(input('what axis u wanna see: '))

i = 0

while(1):
     print('position of axis: ', anc.getPosition(val2))
     time.sleep(0.2)
     i=i+1

# print('position of x axis:', anc.getPosition(0))

# val = input('Enter x axis position desired: (e.g. 2950000) ')

# anc.moveAbsolute(0, int(val))