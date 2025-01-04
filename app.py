from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session, flash
import mysql.connector
# from picamera2 import Picamera2
import io
import time
from PIL import Image
import cv2
import dlib
import numpy as np
import threading
import os

# ngrok http http://localhost:5000

app = Flask(__name__)


# Database configuration (replace with your computer's SQL server details)
DB_HOST = "10.100.102.9"  # Replace with the IP address of your computer
DB_USER = "jonahrpi"  # Replace with your SQL username
DB_PASSWORD = "0586889675"  # Replace with your SQL password
DB_NAME = "camproject"  # Replace with the database name
app.secret_key = "imreallytired"
connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)

# Global variables in order to change modes
camera = None
global_frame = None
lock = threading.Lock()
capture_thread = None
current_mode = "short"
running = False
known_face_encodings = []
known_face_labels = []
PROCESS_EVERY_N_FRAMES = 2
current_frame_count = 0

# Dlib models for detecting faces
face_detector = dlib.get_frontal_face_detector()
face_recognition_model = dlib.face_recognition_model_v1("/home/jonahrpi/Desktop/idcamproject/dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("/home/jonahrpi/Desktop/idcamproject/shape_predictor_68_face_landmarks.dat")


class CamInit:
    def reset_state():
        global camera, global_frame, capture_thread, running
        running = False
        if capture_thread and capture_thread.is_alive():
            capture_thread.join()
        if camera and camera.isOpened():
            camera.release()
        camera = None
        global_frame = None

    def start_camera(width,height):

        global camera, running, capture_thread
        # Initialize the camera
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, 30)
        # Start capturing frames
        running = True
        capture_thread = threading.Thread(target=FrameBG.capture_frames, daemon=True)
        capture_thread.start()


class FrameBG:
    def capture_frames():
        global global_frame, running
        while running:
            result = camera.read()  # result is a tuple (success, frame)
            success, frame = result
            if not success:
                break

            with lock:
                global_frame = frame.copy()

    def generate_regular_frames():
        global running, current_frame_count
        CamInit.reset_state()
        CamInit.start_camera(640,480)
        while running:
            with lock:
                if global_frame is None:
                    continue
            frame = global_frame.copy()
            frame_data = cv2.imencode(".jpg", frame)[1].tobytes() #turns the pic data into jpg into bytes for transfering through http
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")


    def generate_frames(mode):
        if mode=="regular":
            return Response(FrameBG.generate_regular_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
        else:
            global running, current_frame_count
            CamInit.reset_state()
            if mode == 'long':
                upsample_factor = 1
                print(mode, "should be long")
                CamInit.start_camera(640,480)
            else:
                CamInit.start_camera(320,240)
                upsample_factor = 0
                print(mode, "should be short")

            while running:
                with lock:
                    if global_frame is None:
                        continue
                    frame = global_frame.copy()

                current_frame_count += 1
                if current_frame_count % PROCESS_EVERY_N_FRAMES != 0:
                    frame_copy=frame.copy()
                    if mode == 'short':
                        frame_copy = cv2.resize(frame_copy, (640, 480), interpolation=cv2.INTER_LINEAR)
                    frame_data = cv2.imencode(".jpg", frame_copy)[1].tobytes() #turns the pic data into jpg into bytes for transfering through http
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")
                    continue

                # face detection part of system
                # converting famre into RBG for fd system and then sending to face detection func
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_detector(rgb_frame, upsample_factor)

                # goes over each face
                for face in faces:
                    shape = shape_predictor(rgb_frame, face)
                    face_encoding = np.array(face_recognition_model.compute_face_descriptor(rgb_frame, shape))

                    if len(known_face_encodings) > 0:
                        distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                        min_distance_index = np.argmin(distances)
                        label = (
                            known_face_labels[min_distance_index]
                            if distances[min_distance_index] < 0.6
                            else "Unknown"
                        )
                    else:
                        label = "Unknown"

                    # Draw face box and label
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Encode the frame as JPEG
                if mode == 'short':
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
                frame_data = cv2.imencode(".jpg", frame)[1].tobytes() #turns the pic data into jpg into bytes for transfering through http
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n")

class DataEdit():
    def add_face_to_database(image, label):
        cursor = connection.cursor()
        image_blob = cv2.imencode(".jpg", image)[1].tobytes() #turns the pic data into jpg into bytes
        query = "INSERT INTO people (name, face_image) VALUES (%s, %s)"
        cursor.execute(query, (label, image_blob))
        connection.commit()
        cursor.close()


    def load_known_faces_from_database():
        global known_face_encodings, known_face_labels
        known_face_encodings = []
        known_face_labels = []

        try:
            cursor = connection.cursor()
            query = "SELECT name, face_image FROM people"
            cursor.execute(query)
            for name, face_blob in cursor.fetchall():
                # Decode the image from the binary blob
                face_image = cv2.imdecode(np.frombuffer(face_blob, np.uint8), cv2.IMREAD_COLOR)
                rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                faces = face_detector(rgb_image, 0)

                if len(faces) > 0:
                    face = faces[0]
                    shape = shape_predictor(rgb_image, face)
                    face_encoding = np.array(face_recognition_model.compute_face_descriptor(rgb_image, shape))
                    known_face_encodings.append(face_encoding)
                    known_face_labels.append(name)

            cursor.close()
        except mysql.connector.Error as err:
            print(f"Error: {err}")


class User:

    def login_check():
        userDetails = request.form
        id = userDetails['id']
        password = userDetails['password']
        cursor = connection.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE id = %s AND password = %s', (id, password,))
        account = cursor.fetchone()
        return account
class Admin(User):
    def insert_user():
        userDetails = request.form
        id = userDetails['id']
        name = userDetails['name']
        admin = userDetails.get('admin', 0)
        print(admin)
        password = userDetails['password']
        cursor = connection.cursor(dictionary=True)
        cursor.execute("INSERT INTO users(id, name, password, admin) VALUES(%s, %s, %s, %s)", (id, name, password, admin))
        connection.commit()
        cursor.close()
        flash("User added successfully!", "success")

    def remove_user():
        userDetails = request.form
        id_remove = userDetails['id_remove']
        cursor = connection.cursor()
        #comma is important, dont remove turns it into tuple
        cursor.execute("DELETE FROM users WHERE id = %s", (id_remove,))
        connection.commit()
        cursor.close()

    def all_users():
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM users")
        userDetails = cursor.fetchall()
        return userDetails




@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST' and 'id' in request.form and 'password' in request.form:
        account = User.login_check()
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']  # Use the key 'id' from the dictionary
            session['name'] = account['name']  # Use the key 'name' from the dictionary
            session['admin'] = account['admin']  # Use the key 'admin' from the dictionary
           # session['id'] = account['id']
            # Redirect to home page
            return redirect('/main_page')
        else:
            flash('Incorrect username or password!', 'error')
            print("Flashing message: Incorrect username or password!")  # Debugging
    return render_template('login.html')

@app.route('/main_page')
def users():
    if 'loggedin' not in session:
        return redirect('/')
    global current_mode
    return render_template('main_page.html', current_mode=current_mode)

@app.route("/switch_mode/<mode>")
def switch_mode(mode):
    global current_mode
    CamInit.reset_state()
    current_mode = mode
    return render_template('main_page.html', current_mode=current_mode)

@app.route("/video_feed/<mode>")
def video_feed(mode):
    #response- send the follwing to client, mimetype- type of message
    #multipart/x-mixed-replace;- type of mime to send one piece of data at a time where one piece of data replaces the next so it looks like a video
    #boundary=frame- comes from the --frame from above it shows where each section of data ends
    if mode=="regular":
        return Response(FrameBG.generate_regular_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    return Response(FrameBG.generate_frames(mode), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/add_face", methods=["POST"])
def add_face():
    name = request.form.get("name")
    if not name:
        return "Name is required", 400

    with lock:
        if global_frame is None:
            return "No frame captured", 400
        frame = global_frame.copy()

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_frame, 0)

    if len(faces) == 0:
        return "No face detected", 400

    face = faces[0]
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    face_image = frame[y:y+h, x:x+w]
    shape = shape_predictor(rgb_frame, face)
    face_encoding = np.array(face_recognition_model.compute_face_descriptor(rgb_frame, shape))

    # Save to database
    DataEdit.add_face_to_database(face_image, name)

    # Update in-memory known faces
    known_face_encodings.append(face_encoding)
    known_face_labels.append(name)
    global current_mode
    if not current_mode:
        current_mode = "short"  # Default mode
    return redirect(url_for("main_page.html", current_mode=current_mode))

@app.route('/admin_page', methods=['GET', 'POST'])
def admin_page():
    if 'loggedin' not in session:
        return redirect('/')
    if  session['admin'] ==0:
        return redirect('/main_page')
    # stop_camera()
    if request.method == 'POST' and 'id' in request.form and 'password' in request.form :
        # 
        Admin.insert_user()
        return redirect('/admin_page')
    
    if request.method == 'POST' and 'id_remove' in request.form:
        # 
        Admin.remove_user()
        return redirect('/admin_page')
    
    userDetails = Admin.all_users()
    return render_template('admin_page.html',userDetails=userDetails)

@app.route('/logout')
def logout():
    # stop_camera()
    # Clear the session data
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('name', None)
    session.pop('admin', None)
    return redirect('/')

if __name__ == "__main__":
    # Load known faces and start the application
    DataEdit.load_known_faces_from_database()
    # start_camera()
    app.run(host="0.0.0.0", port=5000, threaded=True)


# CREATE USER 'jonahrpi'@'10.100.102.23' IDENTIFIED BY '0586889675';
# GRANT ALL PRIVILEGES ON camproject.* TO 'jonahrpi'@'10.100.102.23';
# FLUSH PRIVILEGES;
