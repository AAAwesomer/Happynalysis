import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Setting font
font = cv2.FONT_HERSHEY_SIMPLEX

# Defining the function that makes decections
def detect(gray, frame):
    happy = False

    # Detecting faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:

        # Creating rectangle surrounding face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Regions of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detecting eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 20)
        for (ex, ey, ew, eh) in eyes:
            # Creating rectangles surrounding eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detecting smiles
        smile = smile_cascade.detectMultiScale(roi_gray, 2.4, 28)
        for (sx, sy, sw, sh) in smile:
            # Creating rectangles surrounding eyes
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
            happy = True

    # Writing text based on detections
    sentiment_text = "Person is {}".format("happy!" if happy else "neutral.")
    cv2.putText(frame, sentiment_text, (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    return frame

video = cv2.VideoCapture(0)

# Until user presses 'q', video from the webcam is read and detections are drawn onto the frames.
while True:
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


