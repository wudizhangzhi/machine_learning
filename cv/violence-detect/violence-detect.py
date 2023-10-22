import onnxruntime as ort
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


WIDHT = HEIGHT = 224
model_path = "/Users/zhangzhichao/ML/models/real-life-violence.onnx"
classes = {0: "NonViolence", 1: "Violence"}

model = YOLO(model_path)
session = ort.InferenceSession(
    model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
# Get the model inputs
model_inputs = session.get_inputs()


def detect(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_gray = cv2.equalizeHist(frame_gray)
    # # Resize the image to match the input shape
    image_data = cv2.resize(img, (WIDHT, HEIGHT))
    # # Normalize the image data by dividing it by 255.0
    # image_data = np.array(face) / 255.0
    # # Transpose the image to have the channel dimension as the first dimension
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

    # # Expand the dimensions of the image data to match the expected input shape
    image_data = np.expand_dims(image_data, axis=0).astype(
        np.float32
    )  # Channel first
    outputs = session.run(None, {model_inputs[0].name: image_data})
    classes_scores = np.transpose(np.squeeze(outputs[0]))
    score = np.amax(classes_scores)
    class_id = np.argmax(classes_scores)
    # box = (x, y, w, h)
    # results = model.predict(image_data)
    # result = results[0]
    # probs = result.probs  # Probs object for classification outputs

    # print(probs)
    # draw_detections(frame, box, score, class_id)
    label = f"{classes[class_id]}: {score:.2f}"
    print(label, end='\r')

    # Calculate the dimensions of the label text
    (label_width, label_height), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.putText(
        frame,
        label,
        (0, frame.shape[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
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
    output_img = detect(frame)
    cv2.imshow("Detected Objects", output_img)
