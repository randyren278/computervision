import os
import datetime
import pickle
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import util

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x520+350+100")

        # Create a label to display the webcam feed
        self.webcam_label = tk.Label(root)
        self.webcam_label.pack()

        # Create buttons for login, logout, and register new user
        self.login_button = util.get_button(root, 'Login', 'green', self.login)
        self.login_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.logout_button = util.get_button(root, 'Logout', 'red', self.logout)
        self.logout_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.register_button = util.get_button(root, 'Register New User', 'gray', self.register_new_user, fg='black')
        self.register_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Set up directories
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Failed to open webcam.")
            self.root.destroy()
            return

        # Load known faces
        self.known_face_encodings, self.known_face_names = self.load_known_faces()

        self.most_recent_capture_arr = None
        self.update_webcam_feed()

    def load_known_faces(self):
        known_face_encodings = []
        known_face_names = []

        for filename in os.listdir(self.db_dir):
            if filename.endswith('.pickle'):
                with open(os.path.join(self.db_dir, filename), 'rb') as file:
                    known_face_encodings.append(pickle.load(file))
                known_face_names.append(filename[:-7])

        return known_face_encodings, known_face_names

    def update_webcam_feed(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = frame[:, :, ::-1]  # Convert from BGR to RGB
            self.most_recent_capture_arr = rgb_frame

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Loop through each face in this frame of video
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.webcam_label.imgtk = imgtk
            self.webcam_label.configure(image=imgtk)

        self.root.after(10, self.update_webcam_feed)

    def login(self):
        if self.most_recent_capture_arr is not None:
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)
            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Oops...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Welcome back!', f'Welcome, {name}.')
                with open(self.log_path, 'a') as f:
                    f.write(f'{name},{datetime.datetime.now()},in\n')

    def logout(self):
        if self.most_recent_capture_arr is not None:
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)
            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Oops...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Goodbye!', f'Goodbye, {name}.')
                with open(self.log_path, 'a') as f:
                    f.write(f'{name},{datetime.datetime.now()},out\n')

    def register_new_user(self):
        register_window = tk.Toplevel(self.root)
        register_window.title("Register New User")
        register_window.geometry("1200x520+370+120")

        capture_label = tk.Label(register_window)
        capture_label.pack()

        if self.most_recent_capture_arr is not None:
            img = Image.fromarray(self.most_recent_capture_arr)
            imgtk = ImageTk.PhotoImage(image=img)
            capture_label.imgtk = imgtk
            capture_label.configure(image=imgtk)

        name_label = util.get_text_label(register_window, "Please, input username:")
        name_label.pack()

        entry_text = util.get_entry_text(register_window)
        entry_text.pack()

        def accept():
            name = entry_text.get(1.0, "end-1c").strip()
            if not name:
                util.msg_box('Error', 'Please enter a valid username.')
                return

            if self.most_recent_capture_arr is not None:
                img_rgb = self.most_recent_capture_arr
                try:
                    embeddings = face_recognition.face_encodings(img_rgb)[0]
                    with open(os.path.join(self.db_dir, f'{name}.pickle'), 'wb') as file:
                        pickle.dump(embeddings, file)
                    util.msg_box('Success!', 'User was registered successfully!')
                    register_window.destroy()
                except IndexError:
                    util.msg_box('Error', 'No face detected. Please try again.')

        accept_button = util.get_button(register_window, 'Accept', 'green', accept)
        accept_button.pack(side=tk.LEFT, padx=10, pady=10)

        try_again_button = util.get_button(register_window, 'Try again', register_window.destroy, bg='red', fg='white')
        try_again_button.pack(side=tk.LEFT, padx=10, pady=10)

    def on_closing(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
