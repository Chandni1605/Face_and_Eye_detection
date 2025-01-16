import cv2
face_cascade = cv2.CascadeClassifier("/Users/chandnisingh/Desktop/Project ds/Haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("/Users/chandnisingh/Desktop/Project ds/Haarcascades/haarcascade_eye.xml")

def detect(input_gray, input_frame):
    faces = face_cascade.detectMultiScale(input_gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = input_gray[y:y+h, x:x+w]
        roi_color = input_frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame
# doing some Face recognition with webcam

video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow("Video", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


