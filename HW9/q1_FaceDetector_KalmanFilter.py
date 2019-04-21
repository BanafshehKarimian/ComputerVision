import  cv2
import numpy as np
Kalman_state_hidden = 4
Kalman_measurement = 2
Kalman_control = 0


def Kalman(v):
    #get one Frame
	ret, frame = v.read()
    if ret == False:
        return
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #first detect faces
	faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        c, r, w, h = (0, 0, 0, 0)
    else:
        c, r, w, h = faces[0]

	#initialization
    kalman = cv2.KalmanFilter(Kalman_state_hidden, Kalman_measurement, Kalman_control) 
    kalman.transitionMatrix = np.array([[1., 0., 0.1, 0.],[0., 1., 0., 0.1],[0., 0., 1., 0.],[0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4) 
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4) 
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64') 
    
	while (1):
		# read the frame
        ret, frame = v.read() 
        if ret == False:
            break
		#predict using kalman.predict()
        prediction = kalman.predict()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		# detect faces for measurement step
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        if len(faces) == 0:
            c, r, w, h = (0, 0, 0, 0)
        else:
            c, r, w, h = faces[0]
        measurement = np.array([c + w / 2, r + h / 2], dtype='float64')
		#initialize final
        final = prediction
		#if faces measured: 
        if c != 0 and w != 0 and r != 0 and h != 0:
			#correct the detection
            posterior = kalman.correct(measurement)
            final = posterior
        #draw rectangle around faces
		(x,y,z,q) = final
        cv2.rectangle(frame, ((int)(x-h/2), (int)(y-w/2)), ((int)(x+h/2), (int)(y+w/2)), (0, 255, 0), 2)
        cv2.imshow('img', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
video_capture = cv2.VideoCapture(0)
Kalman(video_capture)
video_capture.release()
cv2.destroyAllWindows()
