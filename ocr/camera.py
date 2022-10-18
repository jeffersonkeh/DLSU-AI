from threading import Thread
import cv2
import time
class BaseCamera(object):
    def __init__(self, threadname="VideoStream"):
        
        # Initialize thread
        self.thread = Thread(target=self.update, name=threadname, args=())
        self.thread.daemon = True
        self.thread_process = None
        # Initialize frame
        self.frame = None

        # Initialize thread flag
        self.stopped = False

    def start(self):
        '''Start the frame grab thread'''
        # raise RuntimeError('Must be implemented by subclasses.')
        self.thread.start()
        time.sleep(1)
        return self

    def update(self):
        '''Continuously Grab Next frame from the camera'''
        raise RuntimeError('Must be implemented by subclasses.')
        

    def read(self):
        '''Return the next frame from the queue to the user'''
        try:
            return self.frame
        except:
            raise RuntimeError('Must implement Update Method by subclasses.')
        
    
    def stop(self):
        '''Stop the frame grab thread'''
        self.stopped = True
        self.thread.join()

class OpencvCamera(BaseCamera):
    def __init__(self, src, threadname="VideoStream"):
        super(OpencvCamera, self).__init__(threadname)
        
        #initialize the camera thread
        self.stream = cv2.VideoCapture(src)
        
        #read the first frame
        (self.iscaptured, self.frame) = self.stream.read()

    def start(self):
        '''Start the frame grab thread'''
        return super(OpencvCamera, self).start()

    def update(self):
        '''Continuously Grab Next frame from the camera'''
        while True:

            # Stop the thread when the stop method is triggered
            if self.stopped:
                return 

            #Get next frame
            (self.iscaptured, self.frame) = self.stream.read()

        time.sleep(0.1)
        self.stream.release()

    def read(self):
        '''Return the next frame from the queue to the user'''
        return super(OpencvCamera, self).read()

    def stop(self):
        '''Stop the frame grab thread'''
        super(OpencvCamera, self).stop()

        