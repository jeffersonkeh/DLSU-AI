from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# note check "ls /dev/video" to get the list of webcams
camera = cv2.VideoCapture("/dev/video0")  # "/dev/video0 for laptop" "/dev/video4 for external"
camera2 = cv2.VideoCapture("/dev/video2")  # "/dev/video0 for laptop" "/dev/video4 for external"

def gen_frames(camera=camera):  # generate frame by frame from camera
    count=0
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            if count == 1:
                count = 0;
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        count+=1

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(camera2), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('main.html')

@app.route('/vid1')
def vid1():
    """Video streaming home page."""
    return render_template('camera1.html')

@app.route('/maps1')
def maps1():
    """Video streaming home page."""
    return render_template('maps1.html')

@app.route('/vid2')
def vid2():
    """Video streaming home page."""
    return render_template('camera2.html')

@app.route('/maps2')
def maps2():
    """Video streaming home page."""
    return render_template('maps2.html')

@app.route('/logs')
def logs():
    return "LOGS"


if __name__ == '__main__':
    app.run(debug=False)