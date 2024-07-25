import os
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import util
from test import test

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
                                                                    self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './log.txt'

        self.cap = None
        self.setup_webcam()

        # Initialize attributes
        self.most_recent_capture_pil = None
        self.most_recent_capture_arr = None
        self.register_new_user_capture = None
        self.entry_text_register_new_user = None

    def setup_webcam(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Use AVFoundation for macOS
        if not self.cap.isOpened():
            print("Failed to open webcam.")
            return

        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()

        if not ret:
            print("Failed to grab frame")
            self.main_window.after(20, self.process_webcam)
            return

        # Convert the captured frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_arr = frame_rgb  # Save the RGB frame

        # Display the frame in the Tkinter window
        self.most_recent_capture_pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self.webcam_label.imgtk = imgtk
        self.webcam_label.configure(image=imgtk)

        self.main_window.after(20, self.process_webcam)

    def login(self):
        label = test(
            image=self.most_recent_capture_arr,
            model_dir='/path/to/your/model/dir',  # Update the path accordingly
            device_id=0
        )

        if label == 1:
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)
            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write('{},{},in\n'.format(name, datetime.datetime.now()))
        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake !')

    def logout(self):
        label = test(
            image=self.most_recent_capture_arr,
            model_dir='/path/to/your/model/dir',  # Update the path accordingly
            device_id=0
        )

        if label == 1:
            name = util.recognize(self.most_recent_capture_arr, self.db_dir)
            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Hasta la vista !', 'Goodbye, {}.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write('{},{},out\n'.format(name, datetime.datetime.now()))
        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake !')

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        if self.most_recent_capture_pil:
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        # Ensure the image is in RGB format
        img_rgb = self.register_new_user_capture
        print("Register new user image shape:", img_rgb.shape)  # Debugging statement

        embeddings = face_recognition.face_encodings(img_rgb)[0]
        with open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb') as file:
            pickle.dump(embeddings, file)
        util.msg_box('Success!', 'User was registered successfully !')
        self.register_new_user_window.destroy()

    def start(self):
        self.main_window.mainloop()

if __name__ == "__main__":
    app = App()
    app.start()
