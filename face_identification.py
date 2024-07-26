import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tkinter as tk
from tkinter import simpledialog
from tkinter import messagebox
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

class NameDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Enter Name")
        self.geometry("300x100")
        
        self.label = tk.Label(self, text="Enter name for the captured face:", fg="white",bg="black")
        self.label.pack(pady=10)
        
        self.entry = tk.Entry(self, fg='white', bg='black') 
        self.entry.pack(pady=5)
        
        self.submit_button = tk.Button(self, text="Submit", command=self.on_submit)
        self.submit_button.pack(pady=5)
        
        self.name = None
        
    def on_submit(self):
        self.name = self.entry.get()
        self.destroy()

def get_name():
    dialog = NameDialog(root)
    root.wait_window(dialog)
    return dialog.name

def save_face():
    global cap, encodeListKnown, classNames

    # Capture the current frame from the webcam
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get the name using the custom dialog
        name = get_name()

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

def update_frame():
    success, img = cap.read()
    if success:
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

    root.after(10, update_frame)  # Schedule the function to be called again after 10 ms

# Start updating frames
update_frame()

# Start the tkinter main loop
root.mainloop()

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
