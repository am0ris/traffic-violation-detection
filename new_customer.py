import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
import math

# Initialize YOLO model (make sure the model file path is correct)
model = YOLO(r"D:\my project\Milk _Detection_Counting\yolov8n.pt")
# model.export(format="tflite")
# Open video file (make sure the file path is correct)
cap = cv2.VideoCapture(r"E:\windows\cpv\New folder (3)\Red light jumper...right next to police car_...lol(720P_HD).mp4")

# Image dimensions
width = 800
height = 500


traffic = ["Green", "Red", "Yellow"]

po1_rec=(513 ,250)
po12_rec=(126 ,250)




Traffic_color = ""

# Main loop for object detection
while cap.isOpened():



    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (width, height))
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    cv2.line(frame,(po1_rec),(po12_rec),(255, 0, 0),2)

    # Perform object detection
    result = model(frame, stream=True)
    for i in result:
        boxes = i.boxes
        for box in boxes:
            x, y, w, z = box.xyxy[0]
            x, y, w, z = int(x), int(y), int(w), int(z)
            w = w - x
            h = z - y

            # X,Y of centroid circle
            cx = x + w // 2
            cy = y + h // 2

            cls = int(box.cls[0])


            conf = math.ceil(box.conf[0] * 100) / 100

            # Draw the rectangle







            if cls == 9:  # Adjust class index as needed




                # Draw on black_image

                cvzone.cornerRect(frame, (x, y, w, h), colorR=False, colorC=(0, 0, 0))




                cv2.rectangle(black_image, (x, y), (x + w, y + h), (255, 255, 255), -1)

                # cv2.rectangle(frame, (x, y), (x + w, y + h // 3), (0, 0, 255), 2)
                # cv2.rectangle(frame, (x, y + h // 3), (x + w, y + (h // 3) * 2), (0, 255, 255), 2)
                # cv2.rectangle(frame, (x, y + (h // 3) * 2), (x + w, y + h), (0, 255, 0), 2)


            bitwise = cv2.bitwise_and(frame, black_image)


            # Convert the frame to HSV color space

            hsv_frame = cv2.cvtColor(bitwise, cv2.COLOR_BGR2HSV)


            gray_image = cv2.cvtColor(bitwise, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to the grayscale image
            _, canny_black_image = cv2.threshold(gray_image, 70, 250, cv2.THRESH_BINARY)

            # kernel = np.ones((3, 3), np.uint8)
            #
            # # Apply dilation
            # canny_black_image = cv2.dilate(canny_black_image, kernel)

            if cv2.countNonZero(canny_black_image[y:y + h // 3, x:x + w]) > 20:
                # Show the detected color on the frame
                Traffic_color="Red"
            elif cv2.countNonZero(canny_black_image[y + h // 3:y + (h // 3) * 2, x:x + w]) > 20:
                Traffic_color="Yellow"

            elif cv2.countNonZero(canny_black_image[y + (h // 3) * 2:y + h, x:x + w]) > 20:
                Traffic_color = "Green"

            cv2.putText(frame, f"Traffic Light: {Traffic_color}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 255), 2)


            if cls == 2 :


                if 290 < cy < 310 and 126 < cx < 515 and Traffic_color=="Red":

                    cvzone.cornerRect(frame, (x, y, w, h), colorR=False, colorC=(255, 255, 255))


    # Display frames
    cv2.imshow("frame", frame)
    # cv2.imshow("roi", black_image)
    cv2.imshow("cany_black_image",canny_black_image)


    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()




