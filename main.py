import tkinter as tk
from tkinter import *
import requests
from PIL import ImageTk
from PIL import Image
import json
from getmac import get_mac_address as gma
import os.path

# capture faces imports
import numpy as np
import cv2
from datetime import datetime
import os
from train_model import train_livenss_model

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


protoPath = resource_path('deploy.prototxt')
modelPath = resource_path('res10_300x300_ssd_iter_140000.caffemodel')

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# vs = cv2.VideoCapture('video/34.mp4')
frame_count = 0
db_name = "NHPC.db"


def submitact():
    authorization_key_value = authorizationKey.get()

    system_uuid = gma()
    if authorization_key_value and system_uuid:
        url = "https://www.airface.in/airface_web_service/login_auth_key.php"
        # url = "http://localhost/airface_web_service/login_auth_key.php"
        payload = {'sec_key': str(authorization_key_value),
                   'sys_id': str(system_uuid),
                   'btn_login': ''}
        files = [
        ]
        headers = {
            'Cookie': 'PHPSESSID=sgejc57br9d098d5vuqs8keace'
        }
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
        responce = response.text.encode('utf8')
        responce_data = json.loads(responce)
        category = responce_data['status']

        return_data = 1
        if category == 0:

            local_lic_key = responce_data['messagetxt']
            location_id = responce_data['loc_id']
            local_auth_authkey = authorization_key_value

            if local_lic_key != "":
                return_data = 0
            else:
                return_data = 1


    else:
        return_data = 1
    return return_data


def train_window():

    def back_to_face_capture():
        root.destroy()
        new_window()

    root = tk.Tk()

    # put_encodings()
    screen_width = root.winfo_screenwidth() // 3
    screen_height = root.winfo_screenheight() // 3

    root.title("Airface Office Pro")
    C = Canvas(root, bg="blue", height=screen_height, width=screen_width)
    # Background Image
    image = Image.open(resource_path("deer_decode.jpg"))
    image_resize = image.resize((screen_width, screen_height), Image.ANTIALIAS)
    bg_img = ImageTk.PhotoImage(image_resize)
    root.geometry(f"{screen_width}x{screen_height}")

    panel1 = tk.Label(root, image=bg_img)
    panel1.pack(side="top", fill="both", expand="yes")

    panel1.image = bg_img
    train = tk.Button(root, text="Train", command=train_livenss_model, fg="white", bg="#1d2736",
                      font="MSGothic 12 bold")
    train.place(x=round(screen_width / 4), y=round(screen_height / 2), width=round(screen_width / 6))

    back = tk.Button(root, text="Back", command=back_to_face_capture, fg="white", bg="#1d2736",
                     font="MSGothic 12 bold")

    back.place(x=round(screen_width / 1.8), y=round(screen_height / 2), width=round(screen_width / 6))

    root.resizable(False, False)
    root.mainloop()


def new_window():
    root = tk.Tk()

    # put_encodings()
    screen_width = root.winfo_screenwidth() // 3
    screen_height = root.winfo_screenheight() // 3

    root.title("Liveness Model Training")
    C = Canvas(root, bg="blue", height=screen_height, width=screen_width)
    # Background Image
    image = Image.open(resource_path("deer_decode.jpg"))
    image_resize = image.resize((screen_width, screen_height), Image.ANTIALIAS)
    bg_img = ImageTk.PhotoImage(image_resize)
    root.geometry(f"{screen_width}x{screen_height}")

    panel1 = tk.Label(root, image=bg_img)
    panel1.pack(side="top", fill="both", expand="yes")

    panel1.image = bg_img

    def captureRealFaces():
        vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        global frame_count

        # loop over frames from the video file stream
        while True:
            time_stamp = datetime.now()

            time_stamp_str = time_stamp.strftime('%d-%m-%y.%H-%M-%S-%f')
            # grab the frame from the file
            (grabbed, frame) = vs.read()

            # if frame_count % 20 == 0:

            # grab the frame dimensions and construct a blob from the frame
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # ensure at least one face was found
            if len(detections) > 0:
                # we're making the assumption that each image has only ONE
                # face, so find the bounding box with the largest probability
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # ensure that the detection with the largest probability also
                # means our minimum probability test (thus helping filter out
                # weak detections)
                # if confidence > args["confidence"]:
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face and extract the face ROI
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]

                    # save frames in every 10 frames
                    if frame_count % 5 == 0:

                        # create real folder directory if not exists
                        if not os.path.exists('dataset/real'):
                            os.makedirs('dataset/real')
                        # write the frame to disk
                        p = f"dataset/real/{time_stamp_str}.png"
                        cv2.imwrite(p, face)
                        print("[INFO] saved {} to disk".format(p))

                    # Draw rectangle on face
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)

            # Increasee the frame count by 1
            frame_count += 1
            # draw rectangle
            cv2.putText(frame, "Press 'q' to exit", (h - 450, w - 600),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)



            frame = cv2.resize(frame, (w//2,h//2))
            cv2.imshow('Airface', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # do a bit of cleanup
        vs.release()
        cv2.destroyAllWindows()

    def captureFakeFaces():
        vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        global frame_count

        # loop over frames from the video file stream
        while True:
            time_stamp = datetime.now()
            time_stamp_str = time_stamp.strftime('%d-%m-%y.%H-%M-%S-%f')

            # grab the frame from the file
            (grabbed, frame) = vs.read()

            # if frame_count % 10 == 0:

            # grab the frame dimensions and construct a blob from the frame
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            # ensure at least one face was found
            if len(detections) > 0:
                # we're making the assumption that each image has only ONE
                # face, so find the bounding box with the largest probability
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # ensure that the detection with the largest probability also
                # means our minimum probability test (thus helping filter out
                # weak detections)
                # if confidence > args["confidence"]:
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face and extract the face ROI
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX]

                    # save frames in every 10 frames
                    if frame_count % 5 == 0:

                        # create real folder directory if not exists
                        if not os.path.exists('dataset/fake'):
                            os.makedirs('dataset/fake')
                        # print(saved)
                        p = f"dataset/fake/{time_stamp_str}.png"
                        cv2.imwrite(p, face)
                        print("[INFO] saved {} to disk".format(p))

                    # Draw rectangle on face
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)

            # Increasee the frame count by 1
            frame_count += 1

            cv2.putText(frame, "Press 'q' to exit", (h - 450, w - 600),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)



            frame = cv2.resize(frame, (w // 2, h // 2))
            cv2.imshow('Airface', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # do a bit of cleanup
        vs.release()
        cv2.destroyAllWindows()

    def checkData():
        realDir = "dataset/real"
        faskeDir = "dataset/fake"
        isRealDir = os.path.isdir(realDir)
        isFakeDir = os.path.isdir(faskeDir)

        if isRealDir == True and isFakeDir == True:

            print("check for if real and fake data exists in folder")
            # print(isdir)
            root.destroy()
            train_window()
        else:
            print("No data found")

    # main()
    realFaces = tk.Button(root, text="Real Faces", command=captureRealFaces, fg="white", bg="#1d2736",
                          font="MSGothic 10 bold")
    realFaces.place(x=round(screen_width / 4), y=round(screen_height / 1.5), width=round(screen_width / 6))

    fakeFaces = tk.Button(root, text="Fake Faces", command=captureFakeFaces, fg="white", bg="#1d2736",
                          font="MSGothic 10 bold")
    fakeFaces.place(x=round(screen_width / 1.8), y=round(screen_height / 1.5), width=round(screen_width / 6))

    nextButton = tk.Button(root, text="Next", command=checkData, fg="white", bg="#1d2736",
                           font="MSGothic 10 bold")
    nextButton.place(x=round(screen_width / 2.5), y=round(screen_height / 1.2), width=round(screen_width / 6))

    root.resizable(False, False)
    root.mainloop()


def run_exe():
    return_submit = submitact()
    if return_submit != 1:
        msg = "logging in..."
        root.destroy()
        new_window()
    else:
        msg = "Not valid credentials"
        print("Not Aurozied")

    print(msg)
    return msg


root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

root.title("Liveness Model Training")
C = Canvas(root, bg="blue", height=screen_height, width=screen_width)

# Background Image
image = Image.open(resource_path("deer_decode.jpg"))
image_resize = image.resize((screen_width, screen_height), Image.ANTIALIAS)
bg_img = ImageTk.PhotoImage(image_resize)

root.geometry(f"{screen_width}x{screen_height}")

panel1 = tk.Label(root, image=bg_img)
panel1.pack(side="top", fill="both", expand="yes")

panel1.image = bg_img

# Definging the first row
lblAuthKey = tk.Label(root, text="Authorization Key", font=1)
lblAuthKey.place(x=round(screen_width / 3.502), y=round(screen_height / 2.8))

authorizationKey = tk.Entry(root, width=round(screen_width / 15), borderwidth=3, relief="ridge")
authorizationKey.place(x=round(screen_width / 2.577), y=round(screen_height / 2.8), width=round(screen_width / 4.553))

submitbtn = tk.Button(root, text="Submit", command=run_exe, fg="white", bg="#1d2736", font="MSGothic 12 bold")
submitbtn.place(x=round(screen_width / 2.168), y=round(screen_height / 1.95), width=round(screen_width / 15))

root.resizable(False, False)

root.mainloop()
