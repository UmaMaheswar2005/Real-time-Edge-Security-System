import cv2
import os
import time

ADMIN_NAME = "Mahi_admin"
SAVE_PATH = f"dataset/{ADMIN_NAME}"
if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

cap = cv2.VideoCapture(0)

# 1. CAMERA WARMUP (Solves the "Dark Image" issue)
print("Warming up camera sensor...")
for _ in range(30):
    cap.read()

# 2. FIND STARTING NUMBER (Solves the "Overwriting" issue)
# This looks at existing files so it starts at 31.jpg, 32.jpg, etc.
existing_files = [f for f in os.listdir(SAVE_PATH) if f.endswith('.jpg')]
start_count = len(existing_files)
photos_to_take = 15 # Take 15 at a time so you can swap glasses
count = 0

print(f"Starting at image index: {start_count}")
print(f"Capturing {photos_to_take} images. GET READY!")
time.sleep(2)

while count < photos_to_take:
    ret, frame = cap.read()
    if not ret: break

    # Display the countdown on the screen
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Capturing {count+1}/{photos_to_take}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Capture - Stay Still", display_frame)

    # Save with unique index
    img_name = f"{SAVE_PATH}/{start_count + count}.jpg"
    cv2.imwrite(img_name, frame)
    
    print(f"Saved: {img_name}")
    count += 1
    
    # 3. THE DELAY (Solves the "No Time" issue)
    # This gives you 1.5 seconds to shift your head or pose
    cv2.waitKey(1500) 

cap.release()
cv2.destroyAllWindows()
print("\nBatch complete!")
print("Run the script again after changing/removing glasses to add more!")