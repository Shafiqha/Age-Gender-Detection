import cv2
import torch
from deepface import DeepFace
from PIL import Image
import numpy as np

# Load YOLOv5 model for face detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load pre-trained YOLOv5 model

# Load OpenCV DNN models for age and gender detection
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Model mean values and label lists
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(68-100)']
genderList = ['Male', 'Female']

# Initialize webcam
video = cv2.VideoCapture(0)
padding = 20  # Define padding

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert OpenCV BGR image to RGB (YOLOv5 expects RGB input)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLOv5 face detection
    results = model(img_rgb)
    detections = results.xyxy[0]  # YOLOv5 bounding boxes: [x1, y1, x2, y2, confidence, class_id]

    for detection in detections:
        # Extract bounding box and confidence score
        x1, y1, x2, y2, confidence, class_id = detection.tolist()

        if confidence > 0.5:  # Confidence threshold for face detection
            # Convert bounding box coordinates to integer
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Extract the face region with padding
            face_rgb = img_rgb[max(0, y1-padding):min(y2+padding, frame.shape[0]-1),
                               max(0, x1-padding):min(x2+padding, frame.shape[1]-1)]

            # Analyze gender using DeepFace
            try:
                face_np = np.array(face_rgb)  # Ensure it's a NumPy array
                gender_prediction = DeepFace.analyze(face_np, actions=['gender'], enforce_detection=False)
                gender = gender_prediction[0]['gender']  # Access the first element and then get the gender key
            except Exception as e:
                gender = "Unknown"
                print(f"Error analyzing face with DeepFace: {e}")

            # Analyze age using OpenCV DNN
            try:
                face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
                blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                # Predict gender using OpenCV model
                genderNet.setInput(blob)
                genderPred = genderNet.forward()
                gender_opencv = genderList[genderPred[0].argmax()]

                # Predict age
                ageNet.setInput(blob)
                agePred = ageNet.forward()
                age = ageList[agePred[0].argmax()]

                # Use only OpenCV gender prediction and age
                label = f"{gender_opencv}, {age}"
            except Exception as e:
                label = "Age/Gender Error"
                print(f"Error in age/gender prediction: {e}")

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame with bounding boxes and labels
    cv2.imshow("Age-Gender Detection", frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
video.release()
cv2.destroyAllWindows()
