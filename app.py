from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# note check "ls /dev/video" to get the list of webcams
camera = cv2.VideoCapture("/dev/video0")  # "/dev/video0 for laptop" "/dev/video4 for external"
camera2 = cv2.VideoCapture("/dev/video4")  # "/dev/video0 for laptop" "/dev/video4 for external"
success, frame = camera.read() 


def gen_frames(camera=camera):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


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

@app.route('/test_vid')
def test_vid():
    """Video streaming home page."""
    return render_template('test_vid.html')

@app.route('/test_vid2')
def test_vid2():
    """Video streaming home page."""
    return render_template('test_vid2.html')

@app.route('/logs')
def logs():
    return "LOGS"


if __name__ == '__main__':
    app.run(debug=True)