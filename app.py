from flask import Flask, render_template, Response, flash, redirect, url_for
import ocr1
import time
import cv2
import numpy as np

app = Flask(__name__)
app.secret_key = '12345'

hot_car_list = ["ABCD1234"]

# note check "ls /dev/video" to get the list of webcams
camera = cv2.VideoCapture("/dev/video0")  # "/dev/video0 for laptop" "/dev/video4 for external"
camera2 = cv2.VideoCapture("/dev/video2")  # "/dev/video0 for laptop" "/dev/video4 for external"
demo = ocr1.LiveDemo(src=0).start()
time.sleep(1)

count=0
cropped_frame=np.zeros((720,1280,3), np.uint8)
current_plate_num="ABCD1234"

def startDemo():
#     global current_plate_num
    while demo.running():
        frame = demo.read()
        
        if len(demo.plateChars) > 0:
            print(demo.plateChars)
            print(demo.Bbox.rrLocationOfPlateInScene)
#             current_plate_num=demo.plateChars

#             if current_plate_num in hot_car_list:
#                 redirect(url_for('alert'))

            if current_plate_num in hot_car_list:
                redirect(url_for('alert'))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 

        '''
        k = cv2.waitKey(1) & 0xFF
        if k==27:    # Esc key to stop
            print("Exit requested.")
            break
        '''

    demo.stop()

def gen_frames(camera=camera):  # generate frame by frame from camera
    global count
    global cropped_frame
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            if count % 1 == 0:
                # count = 0;
                cropped_frame = frame[1:100, 1:300, :]
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        count+=1

def get_cropped_frame():
    global cropped_frame
    ret, buffer = cv2.imencode('.jpg', cropped_frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/alert')
def alert():
    return render_template('alert.html')

@app.route('/cropped_feed')
def cropped_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(get_cropped_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    #return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(startDemo(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed2')
# def video_feed2():
#     #Video streaming route. Put this in the src attribute of an img tag
#     return Response(gen_frames(camera2), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/crop1')
def crop1():
    """Video streaming home page."""
    return render_template('camera1_cropped.html')

@app.route('/vid1')
def vid1():
    """Video streaming home page."""
    return render_template('camera1.html',current_plate_num=current_plate_num)

@app.route('/maps1')
def maps1():
    """Video streaming home page."""
    return render_template('maps1.html')

# @app.route('/vid2')
# def vid2():
#     """Video streaming home page."""
#     return render_template('camera2.html')

# @app.route('/maps2')
# def maps2():
#     """Video streaming home page."""
#     return render_template('maps2.html')

# @app.route('/logs')
# def logs():
#     return "LOGS"


if __name__ == '__main__':
    app.run(debug=False)