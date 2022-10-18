#!flask/bin/python
import time
from flask import jsonify
from flask import Flask, render_template, Response
import cv2
from camera.Camera import LiveStream

app = Flask(__name__)

vid = LiveStream("webcam")

vid.start()
time.sleep(1)


def gen_frames(camera):  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        frame = camera.read()  # read the camera frame
        ret, buffer = cv2.imencode('.jpg', frame['image'])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed', methods=['GET'])
def camera_stream():
    # with LiveStream("webcam") as vid:
    #     time.sleep(1)
    return Response(gen_frames(vid), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test_vid')
def test_vid():
    """Video streaming home page."""
    return render_template('test_vid.html')


if __name__ == '__main__':
    app.run(debug=False, threaded=True)
    vid.stop()
