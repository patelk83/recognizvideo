import face_recognition
import os
import cv2
import playsound
import speech_recognition as sr 
import pyttsx3
import datetime
import pandas as pd

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


FACESDIR = 'images'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  
video= cv2.VideoCapture(0)


def name_to_color(name):
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color


print('Loading...')
known_faces = []
known_names = []
df=pd.read_excel("data.xlsx")

for name in os.listdir(FACESDIR):

    
    for filename in os.listdir(f'{FACESDIR}/{name}'):       
        image = face_recognition.load_image_file(f'{FACESDIR}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]        
        known_faces.append(encoding)
        known_names.append(name)


while True:
    ret,image =video.read()
    locations = face_recognition.face_locations(image, model=MODEL)
    
    encodings = face_recognition.face_encodings(image, locations)

    print(f', found {len(encodings)} face(s)')
    for face_encoding, face_location in zip(encodings, locations):

       
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

        
        match = None

        if True in results: 
            match = known_names[results.index(True)]
            print(f' - {match} from {results}')
           
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])            
            color = name_to_color(match)           
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)           
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)           
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)           
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)            
            today=datetime.datetime.now().strftime("%x")   
                  
            print(f"Detected {match} at {today}")
            for index,item in df.iterrows():
                if(match in item['Name'] and today not in str(item['Time'])):
                    speak(f" Hello, how are you?{match} ")
                    t= df.loc[index,'Time']
                    df.loc[index,'Time']= str(t)+' ,'+str(today)
                df.to_excel('data.xlsx', index=False)      
           
    cv2.imshow(filename, image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
   