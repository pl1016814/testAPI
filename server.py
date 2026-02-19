#Version 1.9.2 (replaced lane detection with Constantine's method)
import cv2, numpy as np
from flask import Flask, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

FRAME_W, FRAME_H = 640, 480
BOUNDARY = "frame"

# Your preferred working resolution
PROC_W, PROC_H = 400, 200

def open_camera(indices=(0, 1, 2)):
    for idx in indices:
        cam = cv2.VideoCapture(idx)
        ok, frame = cam.read()
        if cam.isOpened() and ok and frame is not None:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            print(f"Camera {idx} OK")
            return cam
        cam.release()
    raise RuntimeError("No working camera found")

camera = open_camera()

def detect_curved_lines(frame):
    # Resize
    img = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)

    # grayscale, blur, canny
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2.5)
    cannyProcess = cv2.Canny(gray, 135, 145)

    # cleanup stuff
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cannyProcess, connectivity=8)
    H = cannyProcess.shape[0]
    minHeightPx = int(0.35 * H)

    clean = np.zeros_like(cannyProcess)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if h > minHeightPx:
            clean[labels == i] = 255
    cannyProcess = clean

    # polylines workflow
    centerPoints = []
    for y in range(cannyProcess.shape[0]):
        xs = np.where(cannyProcess[y] > 0)[0]
        if xs.size:
            xl = int(np.percentile(xs, 10))
            xr = int(np.percentile(xs, 90))
            if (xr - xl > 50) and (xr - xl < 275):
                centerPoints.append((int((xs[0] + xs[-1]) // 2), y))

    # defend against skeletons
    for i in range(0, len(centerPoints) - 25, 25):
        cv2.line(img, centerPoints[i], centerPoints[i + 25], (0, 255, 0), 2)

    # overlay to put the actual lines along that
    overlay = img.copy()
    overlay[cannyProcess > 0] = (255, 0, 0)
    img = cv2.addWeighted(img, 0.75, overlay, 0.25, 0)

    # scale back up
    return cv2.resize(img, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

def mjpeg_stream(processor=None):
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        if processor:
            frame = processor(frame)
        ok, jpg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (b"--" + BOUNDARY.encode() + b"\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_stream(), mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

@app.route("/video_feed_processed")
def video_feed_processed():
    return Response(mjpeg_stream(detect_curved_lines), mimetype=f"multipart/x-mixed-replace; boundary={BOUNDARY}")

@app.route("/")
def home():
    return ('<h1>Camera Server</h1>'
            '<p>Raw: <a href="/video_feed">/video_feed</a></p>'
            '<p>Processed: <a href="/video_feed_processed">/video_feed_processed</a></p>')

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=8081, threaded=True)
    finally:
        camera.release()
