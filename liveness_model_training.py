# import the necessary packages
import numpy as np
import cv2
from datetime import datetime
import os
from train_model import train_livenss_model

protoPath = 'face_detector/deploy.prototxt'
modelPath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far


# vs = cv2.VideoCapture('video/34.mp4')
frame_count = 0


# saved = 1962

def capture_real_face():
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
                if frame_count % 10 == 0:

                    # create real folder directory if not exists
                    if not os.path.exists('dataset/real'):
                        os.makedirs('dataset/real')
                    # write the frame to disk
                    p = f"dataset/real/{time_stamp_str}.png"
                    cv2.imwrite(p, face)
                    print("[INFO] saved {} to disk".format(p))

        # Increasee the frame count by 1
        frame_count += 1
        # draw rectangle
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # do a bit of cleanup
    vs.release()
    cv2.destroyAllWindows()


def capture_fake_face():
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
                if frame_count % 10 == 0:

                    # create real folder directory if not exists
                    if not os.path.exists('dataset/fake'):
                        os.makedirs('dataset/fake')
                    # print(saved)
                    p = f"dataset/fake/{time_stamp_str}.png"
                    cv2.imwrite(p, face)
                    print("[INFO] saved {} to disk".format(p))

        # Increasee the frame count by 1
        frame_count += 1

        # Draw rectangle on face
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # do a bit of cleanup
    vs.release()
    cv2.destroyAllWindows()


message = """Press 1 to capture Real Faces
Press 2 to capture Fake Faces
Press 3 to train the Liveness Model 
Enter: """

choice = input(message)

if choice == '1':
    capture_real_face()

elif choice == '2':
    capture_fake_face()

elif choice == '3':
    train_livenss_model()
