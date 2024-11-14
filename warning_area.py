import cv2
import numpy as np

def draw_grid(frame, width, height):
    # Vẽ lưới tọa độ
    for x in range(0, 2000, 50):
        frame = cv2.line(frame, (x, 0), (x, height), (128, 128, 128), 1)
        cv2.putText(frame, str(x), (x + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    for y in range(0, 2000, 50):
        frame = cv2.line(frame, (0, y), (width, y), (128, 128, 128), 1)
        cv2.putText(frame, str(y), (2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame

def draw_polygon(frame, points):
    for point in points:
        frame = cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)  # Vẽ điểm
    if len(points) > 1:
        frame = cv2.polylines(frame, [np.int32(points)], False, (255, 0, 0), thickness=2)  # Vẽ đa giác
    return frame

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0