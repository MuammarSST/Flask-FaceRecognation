from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from flask_session import Session

import mysql.connector
import cv2
from PIL import Image
import numpy as np
import os
import time
from datetime import date
 
app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
 
cnt = 0
pause_cnt = 0
justscanned = False
 
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="aspire4730z",
    database="flask_db"
)
mycursor = mydb.cursor()

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/admin')
def admin():
    if not session.get("name"):
        return redirect("/")
    mycursor.execute("select * from user")
    data = mycursor.fetchall()
 
    return render_template('index.html', data=data)

@app.route('/tambah_user')
def tambah_user():
    mycursor.execute("select ifnull(max(user_id) + 1, 100) from user")
    row = mycursor.fetchone()
    id_user = row[0]
    return render_template('tambah_user.html',id_user=int(id_user))

@app.route('/simpan_user', methods=['POST'])
def simpan_user():
    user_id = request.form.get('txt_id')
    username = request.form.get('txt_username')
 
    mycursor.execute("""INSERT INTO `user` (`user_id`, `username`) VALUES
                    ('{}', '{}')""".format(user_id, username))
    mydb.commit()
 
    return redirect(url_for('vfdataset_page', user_id=user_id))

@app.route('/vfdataset_page/<user_id>')
def vfdataset_page(user_id):
    return render_template('gendataset.html', user_id=user_id)
 
@app.route('/vidfeed_dataset/<user_id>')
def vidfeed_dataset(user_id):
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(generate_dataset(user_id), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_dataset(user_id):
    face_classifier = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
 
    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        # scaling factor=1.3
        # Minimum neighbor = 5
 
        if faces is ():
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face
 
    cap = cv2.VideoCapture(0)
 
    count_img = 0
    max_imgid=100
 
    while True:
        ret, img = cap.read()
        if face_cropped(img) is not None:
            count_img += 1
           
            face = cv2.resize(face_cropped(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
 
            file_name_path = "dataset/"+user_id+"-"+ str(count_img) + ".jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(count_img), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
 
            frame = cv2.imencode('.jpg', face)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
 
            if cv2.waitKey(1) == 13  or int(count_img) == int(max_imgid) :
                break
                cap.release()
                cv2.destroyAllWindows()
 

@app.route('/train_classifier/<user_id>')
def train_classifier(user_id):
    dataset_dir = "dataset"
 
    path = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)]
    faces = []
    ids = []
 
    for image in path:
        img = Image.open(image).convert('L');
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[-1].split("-")[0])
 
        faces.append(imageNp)
        ids.append(id)
    ids = np.array(ids)
    
 
    # Train the classifier and save
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.json")
 
    return redirect('/admin')

@app.route('/login_video_feed')
def login_video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(login_face_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')

def login_face_recognition():  
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
 
        global justscanned
        global pause_cnt
 
        pause_cnt += 1
 
        coords = []
 
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            id, pred = clf.predict(gray_image[y:y + h, x:x + w])
            confidence = int(100 * (1 - pred / 300))
 
            if confidence > 70 and not justscanned:
                global cnt
                cnt += 1
 
                n = (100 / 30) * cnt
                # w_filled = (n / 100) * w
                w_filled = (cnt / 30) * w
 
                cv2.putText(img, str(int(n))+' %', (x + 20, y + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
 
                cv2.rectangle(img, (x, y + h + 40), (x + w, y + h + 50), color, 2)
                cv2.rectangle(img, (x, y + h + 40), (x + int(w_filled), y + h + 50), (153, 255, 255), cv2.FILLED)
                
                # print(id)
               
             

 
                if int(cnt) == 30:
                    cnt = 0
 
                    mycursor.execute("select * from user where user_id = " + str(id))
                    
                    row = mycursor.fetchone()
                    pnbr = row[0]
                    pname = row[1]



                    
                    
 
                    cv2.putText(img, pname + ' | ' + pnbr, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (153, 255, 255), 2, cv2.LINE_AA)
                    Session["id"]=str(pnbr)
                    Session["name"]=str(pname)
                    return redirect("/admin")

                    time.sleep(1)
 
                    justscanned = True
                    pause_cnt = 0

                    session["id"]=pnbr
                    session["name"]=pname

 
            else:
                if not justscanned:
                    cv2.putText(img, 'Tidak Diketahui', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(img, ' ', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,cv2.LINE_AA)
 
                if pause_cnt > 80:
                    justscanned = False
 
            coords = [x, y, w, h]
        return coords
 
    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 0), "Face", clf)
        return img
 
    faceCascade = cv2.CascadeClassifier("resources/haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.json")
 
    wCam, hCam = 400, 400
 
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
 
    while True:
        ret, img = cap.read()
        img = recognize(img, clf, faceCascade)
 
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
 
        key = cv2.waitKey(1)
        if key == 27:
            break
 



















if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
