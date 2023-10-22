import cv2
import os


# detectorPaths = {
#     "face": "haarcascade_frontalface_default.xml",
#     "eyes": "haarcascade_eye.xml",
#     "smile": "haarcascade_smile.xml",
# }
# detectors = {}

# for (name, path) in detectorPaths.items():
# 	# load the haar cascade from disk and store it in the detectors
# 	# dictionary
# 	path = os.path.sep.join(["data/haarcascades/", path])
# 	detectors[name] = cv2.CascadeClassifier(path)

face_cascade = cv2.CascadeClassifier()
face_cascade_name = "../assets/haarcascades/haarcascade_frontalface_alt.xml"


def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for x, y, w, h in faces:
        center = (x + w // 2, y + h // 2)
        frame = cv2.ellipse(
            frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 1
        )
        # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # faceROI = frame_gray[y : y + h, x : x + w]
    # -- In each face, detect eyes
    # eyes = eyes_cascade.detectMultiScale(faceROI)
    # for (x2,y2,w2,h2) in eyes:
    # eye_center = (x + x2 + w2//2, y + y2 + h2//2)
    # radius = int(round((w2 + h2)*0.25))
    # frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    cv2.imshow("Capture - Face detection", frame)


if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print("--(!)Error loading face cascade")
    exit(0)

img = cv2.imread("../assets/images/1.jpg")
detectAndDisplay(img)
while True:
    if cv2.waitKey(1) == ord("q"):
        break
