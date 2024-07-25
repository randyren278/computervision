import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk

path = 'ImageDatabase'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

def save_face():
    global cap, encodeListKnown, classNames

    # Capture the current frame from the webcam
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Prompt the user to enter a name
        name = simpledialog.askstring("Input", "Enter name for the captured face:")

        if name:
            # Save the image to the ImageDatabase folder
            img_path = os.path.join(path, f"{name}.jpg")
            cv2.imwrite(img_path, img)

            # Update the list of known faces
            new_image = face_recognition.load_image_file(img_path)
            new_encoding = face_recognition.face_encodings(new_image)[0]
            encodeListKnown.append(new_encoding)
            classNames.append(name)
            print(f"Saved and encoded new face for {name}")

encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize the main application window
root = tk.Tk()
root.title("Face Recognition Attendance")

# Create a button to save face
save_button = tk.Button(root, text="Save Face", command=save_face)
save_button.pack(side=tk.TOP, anchor=tk.NE)

# Initialize the video capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Invert the camera display
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Draw blue box around all detected faces
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            # Draw different green box around recognized face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 128, 0), cv2.FILLED)

            # Change font and put text on recognized face
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(img, name, (x1 + 6, y2 - 6), font, 1, (255, 255, 255), 2)
            markAttendance(name)

    # Display the resulting frame
    cv2.imshow('Webcam', img)

    # Process GUI events
    root.update()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
root.destroy()
