import onnxruntime as ort
import cv2
import numpy as np
import matplotlib.pyplot as plt

WIDHT = HEIGHT = 64
model_path = "/Users/zhangzhichao/ML/models/mask-detection-cls_30min.onnx"
classes = {0: 'mask_weared_incorrect', 1: 'with_mask', 2: 'without_mask'}

session = ort.InferenceSession(
    model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
# Get the model inputs
model_inputs = session.get_inputs()

# Store the shape of the input for later use
# input_shape = model_inputs[0].shape

face_cascade = cv2.CascadeClassifier()
face_cascade_name = "../assets/haarcascades/haarcascade_frontalface_alt.xml"
face_cascade.load(cv2.samples.findFile(face_cascade_name))

color_palette = np.random.uniform(0, 255, size=(len(classes), 3))


def draw_detections(img, box, score, class_id):
    x1, y1, w, h = box
    # Retrieve the color for the class ID
    color = color_palette[class_id]

    # Draw the bounding box on the image
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

    # Create the label text with class name and score
    label = f"{classes[class_id]}: {score:.2f}"

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    # Calculate the position of the label text
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

    # Draw a filled rectangle as the background for the label text
    cv2.rectangle(
        img,
        (label_x, label_y - label_height),
        (label_x + label_width, label_y + label_height),
        color,
        cv2.FILLED,
    )

    # Draw the label text on the image
    cv2.putText(
        img,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def detect_mask(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # -- Detect faces
    faces = face_cascade.detectMultiScale(
        frame_gray, minSize=(30, 30),
    )
    for x, y, w, h in faces:
        face = frame[y : y + h, x : x + w]

        # Resize the image to match the input shape
        face = cv2.resize(face, (WIDHT, HEIGHT))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(face) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        outputs = session.run(None, {model_inputs[0].name: image_data})
        classes_scores = np.transpose(np.squeeze(outputs[0]))
        score = np.amax(classes_scores)
        class_id = np.argmax(classes_scores)

        box = (x, y, w, h)
        draw_detections(frame, box, score, class_id)
    return frame


cap = cv2.VideoCapture("/Users/zhangzhichao/Downloads/坚如磐石.mp4")
# start_time = 5 # skip first {start_time} seconds
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

while cap.isOpened():
    # Press key q to stop
    if cv2.waitKey(1) == ord("q"):
        break
    try:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue
    # Update object localizer
    output_img = detect_mask(frame)
    cv2.imshow("Detected Objects", output_img)
