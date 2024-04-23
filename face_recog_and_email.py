import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image
import smtplib,ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os
from dotenv import load_dotenv
load_dotenv()

# Load the face recognition model
model = load_model(r'D:\Deep Learning Project\Face Recognition Model\face_recognition.h5')
# Load the face classifier
face_classifier = cv2.CascadeClassifier(r'D:\Deep Learning Project\Face Recognition Model\haarcascade_frontalface_default.xml')

# Initialize variables for attendance tracking
attendance = {}  # Dictionary to store attendance status of recognized faces
detected_faces = dict()  # List to store names of detected faces

def send_email(subject, body, to_email):
    # Function to send email
    # Configure email settings (SMTP server, sender's email, etc.)
    smtp_server = 'smtp.gmail.com'  # Update with your SMTP server address
    sender_email = os.environ.get("Sender_Email")  # Update with your email address
    password = os.environ.get("Sender_Password")  # Update with your email password
    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(body, 'plain'))
    msg_string = msg.as_string()
    # Connect to SMTP server and send the email
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, 587) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(sender_email, password)
        server.sendmail(sender_email,to_email,msg_string)

def send_sheet_email(subject, body, to_email):
    # Function to send email
    # Configure email settings (SMTP server, sender's email, etc.)
    smtp_server = 'smtp.gmail.com'  # Update with your SMTP server address
    sender_email = 'devanshumishra543@gmail.com'  # Update with your email address
    password = 'jcziryjtlptrkxxq'  # Update with your email password
    # Create a multipart message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(body, 'plain'))
    with open('attendance_sheet.txt', 'rb') as file:
        attachment = MIMEText(file.read(), 'plain', _charset='utf-8')
        attachment.add_header('Content-Disposition', 'attachment', filename='attendance_sheet.txt')
        msg.attach(attachment)

    msg_string = msg.as_string()
    # Connect to SMTP server and send the email
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_server, 587) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(sender_email, password)
        server.sendmail(sender_email,to_email,msg_string)

def face_extractor(img):
    # Function to extract face from image
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if faces is ():
        return None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_faces = img[y:y + h + 50, x:x + w + 50]

    return cropped_faces

video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()
    face = face_extractor(frame)

    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
        img_array = np.array(im)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)

        name = 'Unknown'
        if pred[0][0] > 0.5:
            name = 'Devanshu'
            email = 'devbjs123@gmail.com'
            detected_faces.update({name:email})
        if pred[0][1] > 0.5:
            name = 'Priyanshu'
            email = 'devanshumishra543@gmail.com'
            detected_faces.update({name:email})

        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # Mark attendance for the recognized person
        attendance[name]='Present'
    else:
        cv2.putText(frame, "No face Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

# Camera turned off, generate attendance sheet
# Generate attendance sheet
with open('attendance_sheet.txt', 'w') as f:
    for person, status in attendance.items():
        f.write(f"{person}: {status}\n")

# Send email with attendance sheet
send_sheet_email('Attendance Sheet', 'Please find the attendance sheet attached.', 'devbjs123@gmail.com')

# Send email for detected faces
for face_name,face_email in detected_faces.items():
   send_email('Face Detected', f'Face detected in camera {face_name}', face_email)
