import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✅ Camera is working! No SegFault.")
    cap.release()
else:
    print("❌ Camera failed to open.")