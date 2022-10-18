'''
ret values:
0 - no plates
1-5 - plate & chars
'''  

import glob
import cv2
import numpy as np
import os
import time
import cv2
import csv
import datetime

from camera import OpencvCamera
from collections import deque
from threading import Thread

dateNow = datetime.datetime.now()
'''
nameCSV = 'Violation-NumberCoding-' + dateNow.strftime("%Y-%m-%d") + '.csv'
try:
    with open(nameCSV, 'w', newline='') as csvfile:
        fieldnames = ['TIME','DATE','OCR','CODING','FILE LOCATION']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
except:
    pass
'''
window_name = "NSTW - Number Coding Live Demo"

class LiveDemo(OpencvCamera):
    # Set the parameters needed
    queue_size=256
    cons = 5 #frame/s

    def __init__(self, threadAI="ObjectDetection", threadstream="VideoStream"):
        
        # Initialize Video Stream Grabbing
        super(LiveDemo, self).__init__(src=0, threadname=threadstream)    
        
        # initialize the queue used to store frames read from
		# the video file
        self.deq = deque(maxlen=self.queue_size)
        # Initialize all variables need for the plate Detection and OCR
        self.framectr = 0
        self.platectr = 0
        self.ret      = 0
        self.codeDay    = ''
        self.plateChars = ''
        self.savedDay   = ''
        self.savedChars = ''
        self.showBox    = ''
        self.violate    = ''

        #initialize Thread
        self.thread_process = Thread(target=self.update_frame, name=threadAI, args=())
        self.thread_process.daemon = True

    def start(self):
        self = super(LiveDemo, self).start()
        time.sleep(1)
        self.thread_process.start()
        return self

    def update_blackframe(self,ret, violate, black):
        if ret in [1,2,3,4,5,6]:
            #platectr   = platectr + 1
            #outname    = 'lpr/codeTest' + '-' + codeDay + '/' + codeDay + '/' + str(framectr) + '-' +  str(platectr) + '-' + str(plateChars) + '.jpg'
            #cv2.imwrite(outname,plate)
            if violate == 'Y':
                '''
                print('Violated NCS')
                currentDT = datetime.datetime.now()
                with open(nameCSV, 'a', newline='') as csvfile:
                    fieldnames = ['TIME','DATE','OCR','CODING','FILE LOCATION']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    time1 = currentDT.strftime("%I:%M:%S %p")
                    date1 = currentDT.strftime("%Y/%m/%d")
                    writer.writerow({'TIME': time1, 'DATE':date1, 'OCR': plateChars, 'CODING':codeDay, 'FILE LOCATION':outname})
                '''
                violateMsg = 'Violated the Number Coding Scheme!'
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(black,violateMsg,(28,400),font,0.9,(0,0,225),2,cv2.LINE_AA)

            if ret < 6:
                self.savedDay   = self.codeDay
                self.savedChars = self.plateChars
        elif ret == 0:
            self.savedDay   = ''
            self.savedChars = ''
            self.showBox    = 'N'
        
        return black


    def update_frame(self):
        # keep looping infinitely
        from lpr import Main, DetectChars

        while True:
			# if the thread indicator variable is set, stop the
			# thread
            if self.stopped:
                break
            
            self.framectr += 1
            black = np.zeros((480, 640, 3), np.uint8)
            # otherwise, ensure the queue has room in it
            
            # read the next frame from the file
            new_frame = super(LiveDemo, self).read()
            if (self.framectr % self.cons == 0):
                    
                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.

                plate, black, self.plateChars, self.codeDay, violate, self.showBox, ret = Main.recognize(new_frame, self.savedChars, self.savedDay, self.showBox)

                black = self.update_blackframe(ret, violate, black)
                # add the frame to the queue
                # self.Q.put(frame)
            else:
                plate = new_frame    
            
            hstack1 = np.hstack((black,cv2.resize(plate,(640,480))))

            self.deq.appendleft(hstack1)

                # self.Q.put(hstack1)

            # else:
            #     time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        while(len(self.deq) <= 0): 
            time.sleep(0.1)

        return self.deq.pop()

    # Insufficient to have consumer use while(more()) which does
	# not take into account if the producer has reached end of
	# file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
		# return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while len(self.deq) == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return len(self.deq) > 0

    def stop(self):
        super(LiveDemo, self).stop()
        self.thread_process.join()
        self.stopped = True





if __name__ == "__main__":

    demo = LiveDemo().start()
    time.sleep(1)


    while demo.running():
        frame = demo.read()


        cv2.imshow(window_name,frame)
        
        k = cv2.waitKey(1) & 0xFF
        if k==27:    # Esc key to stop
            print("Exit requested.")
            break

        # if len(demo.deq()) < 2:
        #     time.sleep(0.001)


    cv2.destroyAllWindows()
    demo.stop()