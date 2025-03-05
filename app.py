
print("=============  ___ ____     ____                  ____            _           _    ============= ")
print("============= |_ _|  _ \   / ___|__ _ _ __ ___   |  _ \ _ __ ___ (_) ___  ___| |_  ============= ")
print("=============  | || | | | | |   / _` | '_ ` _ \  | |_) | '__/ _ \| |/ _ \/ __| __| ============= ")
print("=============  | || |_| | | |__| (_| | | | | | | |  __/| | | (_) | |  __/ (__| |_  ============= ")
print("============= |___|____/   \____\__,_|_| |_| |_| |_|   |_|  \___// |\___|\___|\__| ============= ")
print("=============                                                   |__/               ============= ")             
import random
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session, flash
import mysql.connector
import time
from PIL import Image
import cv2
import dlib
import numpy as np
import threading
print("got through imports")
# ngrok http http://localhost:5000

app = Flask(__name__)

DB_HOST = "10.38.22.93" #computer ip 10.100.102.9 or 192.168.195.93
DB_USER = "jonahrpi"  
DB_PASSWORD = "0586889675" 
DB_NAME = "camproject" 
app.secret_key = "imreallytired"
connection = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
print("got through sql")
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
firstface = True
identnames = ["jhon", "sam"]
tess_smg = "yes"
unknowns_fc= 0
last_label= ""
attendance_mode = False
grade_names_list = []
class_names_list = []
class_grade_id_list = []
grade_id_list = []
students_list = []
class_id_list = []
# Dlib models for detecting faces
face_detector = dlib.get_frontal_face_detector()
face_recognition_model = dlib.face_recognition_model_v1("/home/jonahrpi/Desktop/idcamproject/dlib_face_recognition_resnet_model_v1.dat")
shape_predictor = dlib.shape_predictor("/home/jonahrpi/Desktop/idcamproject/shape_predictor_68_face_landmarks.dat")

print("got through dlib recognition model")
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
        time.sleep(2)
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

        global identnames, last_label, unknowns_fc
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
                    
                    if last_label!=label:
                        print("here1")
                        identnames.append(label)
                        last_label=label
                    elif label=="Unknown" and last_label!="Unknown":
                        print("here2")
                        unknowns_fc=unknowns_fc + 1
                        identnames.append(label + "." + str(unknowns_fc))

                    
                    print("face detected")
                    # Draw face box and label
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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

    def remove_face_from_database(label):
        cursor = connection.cursor()
        query = "DELETE FROM people WHERE name = %s"
        cursor.execute(query, (label,))
        connection.commit()
        cursor.close()
        if label in known_face_labels:
            index = known_face_labels.index(label)  # Find the index
            del known_face_encodings[index]  # Remove encoding at index
            del known_face_labels[index]  # Remove name at index


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

    def set_grade_to_database(gname):
        cursor = connection.cursor()
        check_query = "SELECT COUNT(*) FROM grades WHERE grade_name = %s"
        cursor.execute(check_query, (gname,))
        result = cursor.fetchone()
        if result[0] == 0:
            query = "INSERT INTO grades (grade_name) VALUES (%s)"
            cursor.execute(query, (gname,))
            connection.commit()
            print(f"Grade '{gname}' added to the database.")
        else:
            print(f"Grade '{gname}' already exists in the database.")
        cursor.close()

    def remove_grade_from_database(gname):
        cursor = connection.cursor()
        
        # Check if the grade exists
        check_query = "SELECT COUNT(*) FROM grades WHERE grade_name = %s"
        cursor.execute(check_query, (gname,))
        result = cursor.fetchone()
        
        if result[0] > 0:  # Grade exists
            # Delete the grade from the database
            delete_query = "DELETE FROM grades WHERE grade_name = %s"
            cursor.execute(delete_query, (gname,))
            connection.commit()
            print(f"Grade '{gname}' removed from the database.")
        else:
            print(f"Grade '{gname}' does not exist in the database.")
        
        cursor.close()

    def set_class_to_database(cname, gradeid):
        cursor = connection.cursor()
        check_query = "SELECT COUNT(*) FROM classes WHERE class_name = %s"
        cursor.execute(check_query, (cname,))
        result = cursor.fetchone()
        check_query = "SELECT COUNT(*) FROM grades WHERE grade_id = %s"
        cursor.execute(check_query, (gradeid,))
        gresult = cursor.fetchone()
        if result[0] == 0 and gresult[0] == 1:
            query = "INSERT INTO classes (class_name, grade_id) VALUES (%s, %s)"
            cursor.execute(query, (cname, gradeid))
            connection.commit()
            print(f"class '{cname}' added to the database.")
        else:
            print(f"class '{cname}' already exists in the database or or gresult isnt working.")
        cursor.close()

    def remove_class_from_database(cname, gradeid):
        cursor = connection.cursor()
        check_query = "SELECT COUNT(*) FROM classes WHERE class_name = %s"
        cursor.execute(check_query, (cname,))
        result = cursor.fetchone()
        check_query = "SELECT COUNT(*) FROM grades WHERE grade_id = %s"
        cursor.execute(check_query, (gradeid,))
        gresult = cursor.fetchone()
        if result[0] == 1 and gresult[0] == 1:
            delete_query = "DELETE FROM classes WHERE class_name = %s"
            cursor.execute(delete_query, (cname,))
            connection.commit()
            print(f"class '{cname}' removed from the database.")
        else:
            print(f"class '{cname}' doest exists in the database or or gresult isnt working.")
        cursor.close()

    def set_student_to_database(studentname, studentid, student_class_id):
        cursor = connection.cursor()
        query = "INSERT INTO students (name, student_entered_id, pic_uploaded, profile_pic, class_id) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(query, (studentname, studentid, False, None, student_class_id))
        connection.commit()
        print(f"student '{studentname}' added to the database.")
        cursor.close()

    def remove_student_from_database(studentname, studentid, student_class_id):
        cursor = connection.cursor()
        check_query = "SELECT COUNT(*) FROM students WHERE name = %s AND class_id = %s"
        cursor.execute(check_query, (studentname, student_class_id))
        result = cursor.fetchone()
        if (result[0] == 1):
            query = "DELETE FROM students WHERE name = %s AND class_id = %s"
            cursor.execute(query, (studentname, student_class_id))
            connection.commit()
        else:
            check_query = "SELECT COUNT(*) FROM students WHERE student_entered_id = %s AND class_id = %s"
            cursor.execute(check_query, (studentid, student_class_id))
            result = cursor.fetchone()
            if (result[0] == 1):
                query = "DELETE FROM students WHERE student_entered_id = %s AND class_id = %s"
                cursor.execute(query, (studentid, student_class_id))
                connection.commit()
                print(f"student '{studentname}' removed from the database.")
            else:
                print(f"student isnt in database")

        cursor.close()

    def get_students_list():
        global students_list
        cursor = connection.cursor()
        query = "SELECT student_id, name, student_entered_id, pic_uploaded, class_id FROM students"
        cursor.execute(query)
        students_list = cursor.fetchall()
        print(f"{students_list}")
        cursor.close()

    def get_grade_names_list():
        global grade_names_list
        cursor = connection.cursor()
        query = "SELECT grade_name FROM grades"
        cursor.execute(query)
        grade_names_list = [row[0] for row in cursor.fetchall()]
        print(f"Grade names: {grade_names_list}")
        cursor.close()

    def get_class_id_list():
        global class_id_list
        cursor = connection.cursor()
        query = "SELECT class_id FROM classes"
        cursor.execute(query)
        class_id_list = [row[0] for row in cursor.fetchall()]
        print(f"class ids: {class_id_list}")
        cursor.close()

    def get_grade_id_list():
        global grade_id_list
        cursor = connection.cursor()
        query = "SELECT grade_id FROM grades"
        cursor.execute(query)
        grade_id_list = [row[0] for row in cursor.fetchall()]
        print(f"Grade ID list: {grade_id_list}")
        cursor.close()

    def get_class_names_list():
        global class_names_list
        cursor = connection.cursor()
        query = "SELECT class_name FROM classes"
        cursor.execute(query)
        class_names_list = [row[0] for row in cursor.fetchall()]
        print(f"class names: {class_names_list}")
        cursor.close()

    def get_class_grade_id_list():
        global class_grade_id_list
        cursor = connection.cursor()
        query = "SELECT grade_id FROM classes"
        cursor.execute(query)
        class_grade_id_list = [row[0] for row in cursor.fetchall()]
        print(f"class names: {class_grade_id_list}")
        cursor.close()


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

def make_lists():
    print("make_lists() called")
    DataEdit.get_students_list()
    DataEdit.get_grade_names_list()
    DataEdit.get_grade_id_list()
    DataEdit.get_class_names_list()
    DataEdit.get_class_id_list()
    DataEdit.get_class_grade_id_list()

which_page = "login"

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
            flash('Incorrect Username Or Password!', 'error')
            print("Flashing message: Incorrect Username Or Password!")  # Debugging
    return render_template('login.html')

@app.route('/main_page')
def users():
    if 'loggedin' not in session:
        return redirect('/')
    global current_mode, identnames, tess_smg, attendance_mode, known_face_labels, which_page
    which_page = "main"
    return render_template('index.html', current_mode=current_mode, identnames=identnames, tess_smg=tess_smg, attendance_mode=attendance_mode, hoob=known_face_labels[0])

@app.route("/switch_mode/<mode>")
def switch_mode(mode):
    global current_mode, identnames, tess_smg, attendance_mode
    CamInit.reset_state()
    current_mode = mode
    return render_template('index.html', current_mode=current_mode, identnames=identnames, tess_smg=tess_smg, attendance_mode=attendance_mode)

@app.route("/video_feed/<mode>")
def video_feed(mode):
    #response- send the follwing to client, mimetype- type of message
    #multipart/x-mixed-replace;- type of mime to send one piece of data at a time where one piece of data replaces the next so it looks like a video
    #boundary=frame- comes from the --frame from above it shows where each section of data ends
    if mode=="regular":
        return Response(FrameBG.generate_regular_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
    return Response(FrameBG.generate_frames(mode), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/remove_face", methods=["POST"])
def remove_face():
    global current_mode, identnames, tess_smg, attendance_mode
    namer = request.form.get("name")
    if not namer:
        return "Name is required", 400
    DataEdit.remove_face_from_database(namer)
    flash('Face Removed', 'error')
    return render_template('index.html', current_mode=current_mode, identnames=identnames, tess_smg=tess_smg, attendance_mode=attendance_mode)

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
    global current_mode, identnames, tess_smg, attendance_mode
    # Update in-memory known faces
    known_face_encodings.append(face_encoding)
    known_face_labels.append(name)
    flash('Added Face', 'error')
    return render_template('index.html', current_mode=current_mode, identnames=identnames, tess_smg=tess_smg, attendance_mode=attendance_mode)

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
    global which_page
    which_page = "admin"
    userDetails = Admin.all_users()
    return render_template('admin_page.html',userDetails=userDetails)

@app.route('/face_lists')
def face_lists():
    global which_page
    which_page = "face_lists"
    return render_template('face_groups.html')

@app.route('/facesseen')
def facesseen():
    print("works")
    if 'loggedin' not in session:
        return redirect('/')
    global current_mode, identnames, tess_smg, attendance_mode
    print(f"Route '/facesseen' was called: {request.method} - {request.url}")
    print("goes through here")
    return render_template('index.html', current_mode=current_mode, identnames=identnames, tess_smg=tess_smg, attendance_mode=attendance_mode)

@app.route('/attendance')
def attendance():
    print("attendance works")
    if 'loggedin' not in session:
        return redirect('/')
    global current_mode, identnames, tess_smg, attendance_mode
    attendance_mode = True
    print(f"Route '/attendance' was called: {request.method} - {request.url}")
    print("goes through here")
    return render_template('index.html', current_mode=current_mode, identnames=identnames, tess_smg=tess_smg, attendance_mode=attendance_mode)

@app.route('/add_grade', methods=['GET', 'POST'])
def add_grade():
    global grade_names_list
    gname = request.form.get("gradename")
    if not gname:
        return "Name is required", 400
    DataEdit.set_grade_to_database(gname)
    make_lists()

    return redirect('/face_lists')
@app.route('/remove_grade', methods=['GET', 'POST'])
def remove_grade():
    global grade_names_list
    gname = request.form.get("gradename")
    if not gname:
        return "Name is required", 400
    DataEdit.remove_grade_from_database(gname)
    make_lists()

    return redirect('/face_lists')

@app.route('/add_class', methods=['GET', 'POST'])
def add_class():
    cname = request.form.get("classname")
    gradeid= request.form.get("gradesection")
    if not cname:
        return "Name is required", 400
    DataEdit.set_class_to_database(cname, gradeid)
    make_lists()
    return redirect('/face_lists')

@app.route('/remove_class', methods=['GET', 'POST'])
def remove_class():
    cname = request.form.get("classname")
    gradeid= request.form.get("gradesection")
    if not cname:
        return "Name is required", 400
    DataEdit.remove_class_from_database(cname, gradeid)
    print(class_names_list, cname)
    print(class_grade_id_list, gradeid)
    make_lists()
    return redirect('/face_lists')

@app.route('/add_student', methods=['GET', 'POST'])
def add_student():
    studentname = request.form.get("studentname")
    studentid = request.form.get("studentid")
    student_class_id = request.form.get("classid")
    if not studentname or not studentid:
        return "Name is required", 400
    DataEdit.set_student_to_database(studentname, studentid, student_class_id)
    make_lists()
    return redirect('/face_lists')

@app.route('/remove_student', methods=['GET', 'POST'])
def remove_student():
    studentname = None
    studentid = None
    student_class_id = None

    studentname = request.form.get("studentname")
    studentid = request.form.get("studentid")
    student_class_id = request.form.get("classid")
    if not studentname and not studentid:
        return "Name is required", 400
    print(f"student: {studentname, studentid, student_class_id}")
    DataEdit.remove_student_from_database(studentname, studentid, student_class_id)
    make_lists()
    return redirect('/face_lists')


@app.route('/logout')
def logout():
    # stop_camera()
    # Clear the session data
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('name', None)
    session.pop('admin', None)
    return redirect('/')

@app.route('/dataupdate')
def api_datapoint():
    global identnames, known_face_labels
    random_number = random.randint(1, 100)
    attendance_mode = True
    
    dictionary_to_return = {
        'random_number': random_number,
        'attendance_mode': attendance_mode,
        'identnames': identnames,
        'facesdb': known_face_labels,
        'grade_names_list': grade_names_list,
        'grade_id_list': grade_id_list,
        'class_names_list': class_names_list,
        'class_grade_id_list': class_grade_id_list,
        'students_list': students_list,
        'class_id_list': class_id_list,
        'which_page': which_page

    }

    return jsonify(dictionary_to_return)



if __name__ == "__main__":
    # Load known faces and start the application
    DataEdit.load_known_faces_from_database()
    make_lists()
    # start_camera()
    app.run(host="0.0.0.0", port=5000, threaded=True)


# CREATE USER 'jonahrpi'@'192.168.195.93' IDENTIFIED BY '0586889675';
# GRANT ALL PRIVILEGES ON camproject.* TO 'jonahrpi'@'192.168.195.93';
# FLUSH PRIVILEGES;

# CREATE USER 'jonahrpi'@'10.100.102.23' IDENTIFIED BY '0586889675';
# GRANT ALL PRIVILEGES ON camproject.* TO 'jonahrpi'@'10.100.102.23';
# FLUSH PRIVILEGES;
