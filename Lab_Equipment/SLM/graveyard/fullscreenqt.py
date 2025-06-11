
import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QCoreApplication
import numpy as np
import time


class FullscreenWindow(QtWidgets.QWidget):
    def __init__(self, screen=0, parent=None):
        """Fullscreen widget that draws a numpy array to the screen"""
        QtWidgets.QWidget.__init__(self,parent) 

        qdw = QtWidgets.QDesktopWidget()
        rect = qdw.screenGeometry(screen=screen)    
        print("Display size (Y,X): ",rect.height(),rect.width())
        self.setGeometry(rect)
    
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        qimg = QtGui.QImage(rect.width(), rect.height(), QtGui.QImage.Format_RGB888)
        self.qimg = qimg

        self.data = self.getBuffer()

        self.rect= rect
                
        self.show()

    def setMonoColorMap(self,cmap):
        qimg = QtGui.QImage(self.rect.width(), self.rect.height(), QtGui.QImage.Format_Indexed8)
        for i,color in enumerate(cmap):
            qimg.setColor(i,color)#QtGui.QColor(color))
        #qimg.setMonoColorMap()

        self.qimg = qimg
        self.data = self.getBuffer()


    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawImage(event, qp)
        qp.end()

    def drawImage(self,event,qp):
        qp.drawImage(0,0,self.qimg)


    def update(self):  
        self.repaint()
        # Daniel 14/09/2023 needed to add the processEvent() to ensure the repaint() is finished before moving to next displayed image.
        # if you dont do this Qt will sometime not write the new image as it thinks you are doing something stupid and will wait until
        # last repaint() is done.
        QCoreApplication.processEvents()
    
    def getBuffer(self):
        img = self.qimg
        ptr = self.qimg.bits()
        ptr.setsize(self.qimg.byteCount())

        
        ## view the data as a writable numpy array
        arr = np.asarray(ptr).reshape(img.height(), img.width(), self.qimg.depth()//8)
        return arr

if __name__ == '__main__':
    
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    fs = FullscreenWindow(screen=1)

    arraytoedit = fs.data

    ## anytime you edit this array, just call fs.update() to display it.
    arraytoedit[:,1920//2:,0] = 255
    fs.update()

    app.exec_()

        

