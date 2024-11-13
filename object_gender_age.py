import cv2
import os
import supervision as sv
from ultralytics import YOLO
import typer

# Load the model
model_yolov10 = YOLO("config/yolov10x.pt")
app = typer.Typer()

category_dict = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
    22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
    27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
    32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
    40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
    77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

def initialize_age_gender_models():
    faceProto = "config/opencv_face_detector.pbtxt"
    faceModel = "config/opencv_face_detector_uint8.pb"
    ageProto = "config/age_deploy.prototxt"
    ageModel = "config/age_net.caffemodel"
    genderProto = "config/gender_deploy.prototxt"
    genderModel = "config/gender_net.caffemodel"

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    
    return faceNet, ageNet, genderNet

def detect_age_gender(image, box, faceNet, ageNet, genderNet, padding=20):
    face = image[
        max(0, int(box[1]) - padding):min(int(box[3]) + padding, image.shape[0] - 1),
        max(0, int(box[0]) - padding):min(int(box[2]) + padding, image.shape[1] - 1)
    ]
    
    if face.size == 0:
        return None, None
    
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    
    # Dự đoán giới tính
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    
    # Dự đoán tuổi
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    
    return gender, age

def detect_faces(faceNet, frame, conf_threshold=0.7):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    faceNet.setInput(blob)
    detections = faceNet.forward()
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            boxes.append([x1, y1, x2, y2])
    return boxes

def process_video_object(video_source, ban_object=["knife"], ban_ages=None, ban_genders=None):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise ValueError("Unable to open video source")

    faceNet, ageNet, genderNet = initialize_age_gender_models()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolov10(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            class_name = category_dict[class_id]
            color_set = (238, 238, 175)

            if class_name == "person":
                color_set = (0, 215, 255)
                faceBoxes = detect_faces(faceNet, frame)
                
                for face_box in faceBoxes:
                    gender, age = detect_age_gender(frame, face_box, faceNet, ageNet, genderNet)
                    
                    # Check if age or gender is in the banned lists
                    if (gender in ban_genders) or (age in ban_ages):
                        color_set = (0, 0, 255)
                        cv2.rectangle(frame, (int(face_box[0]), int(face_box[1])), 
                                      (int(face_box[2]), int(face_box[3])), color_set, 2)
                        cv2.putText(frame, f"{gender}, {age}: {confidence:.2f}", 
                                    (int(face_box[0]), int(face_box[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_set, 2)
                    else:
                        cv2.rectangle(frame, (int(face_box[0]), int(face_box[1])), 
                                      (int(face_box[2]), int(face_box[3])), color_set, 2)
                        cv2.putText(frame, f"{gender}, {age}: {confidence:.2f}", 
                                    (int(face_box[0]), int(face_box[1] - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_set, 2)
            
            elif class_name in ban_object:
                color_set = (0, 0, 255)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), color_set, 2)
                cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                            (int(box[0]), int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_set, 2)
            
            else:
                cv2.rectangle(frame, (int(box[0]), int(box[1])), 
                              (int(box[2]), int(box[3])), color_set, 2)
                cv2.putText(frame, f"{class_name}: {confidence:.2f}",
                            (int(box[0]), int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_set, 2)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

def process_webcam():
    cap = cv2.VideoCapture("http://192.168.1.3:8080/video")  # 0 is typically the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    faceNet, ageNet, genderNet = initialize_age_gender_models()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model_yolov10(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            class_name = category_dict[class_id]
            color_set = (238, 238, 175)
            if class_name == "person":
                color_set = (0, 215, 255)
                # Phát hiện khuôn mặt
                faceBoxes = detect_faces(faceNet, frame)
                
                # Dự đoán tuổi và giới tính cho từng khuôn mặt
                for box in faceBoxes:
                    gender, age = detect_age_gender(frame, box, faceNet, ageNet, genderNet)
                    
                    # Hiển thị thông tin lên khung hình
                    if gender and age:
                        # label = f'{gender}, {age}'
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_set, 2)
                        cv2.putText(frame, f"{gender}, {age}: {confidence:.2f}", (int(box[0]), int(box[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)
                    else:
                        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_set, 2)
                        cv2.putText(frame, f"{class_name}: {confidence:.2f}", (int(box[0]), int(box[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2)
            elif class_name in ["baseball bat", "fork", "knife", "scissors"]:
                color_set = (0, 0, 255)
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_set, 2)
                cv2.putText(frame, f"{class_name}: {confidence:.2f}", (int(box[0]), int(box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_set, 2)
            else:        
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color_set, 2)
                cv2.putText(frame, f"{class_name}: {confidence:.2f}", (int(box[0]), int(box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_set, 2)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.command()
def webcam():
    typer.echo("Starting webcam processing...")
    process_webcam()

if __name__ == "__main__":
    app()