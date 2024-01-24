from flask import Flask, Response, redirect, render_template, request, url_for
from flask_cors import CORS
import cv2
import os

import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('pagina1'))

@app.route('/face-reconocimiento-video')
def pagina1():
    return render_template('video_recognition.html')

@app.route('/record-video', methods=['GET', 'POST'])
def pagina3():
    if request.method == 'POST':
        nombre = request.form['nombre']
        video_path = f'./Videos/{nombre}.mp4'

        # Verificar si ya existe un video con ese nombre
        i = 1
        while os.path.exists(video_path):
            video_path = f'./Videos/{nombre}_{i}.mp4'
            i += 1

        # Iniciar la grabación del video (duración aproximada de 30 segundos)
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

        tiempo_inicial = cv2.getTickCount()
        while (cv2.getTickCount() - tiempo_inicial) / cv2.getTickFrequency() < 30:
            ret, frame = cap.read()
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # return f'Video grabado y guardado en {video_path}'
        
        # Extraer frames y guardar en carpeta con nombre
        person_path = f'./static/Data/{nombre}'
        if not os.path.exists(person_path):
            print('Carpeta creada:', person_path)
            os.makedirs(person_path)

        cap = cv2.VideoCapture(video_path)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        count = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(f'{person_path}/{nombre}_{count}.jpg', face_roi)
                count += 1

        cap.release()
        
        ### Continuar con el entrenamiento del modelo
        dataPath = './static/Data'
        peopleList = os.listdir(dataPath)
        print('Lista de personas: ', peopleList)

        labels = []
        facesData = []
        label = 0

        for nameDir in peopleList:
            personPath = dataPath + '/' + nameDir
            print('Leendo las imágenes de ' + personPath)

            for fileName in os.listdir(personPath):
                labels.append(label)
                facesData.append(cv2.imread(personPath+'/'+fileName, 0))

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Entrenando el reconocedor facial
        print("Entrenando...")
        face_recognizer.train(facesData, np.array(labels))

        # Almacenando el modelo obtenido
        face_recognizer.write('modeloLBPHFace.xml')
        print(f"Modelo almacenado")

    return render_template('grabarvideo.html')

@app.route('/registered-people')
def pagina4():
    # Ruta de tu directorio './Data'
    data_directory = './static/Data'

    # Lista para almacenar la información a presentar en HTML
    folder_info = []

    # Recorre cada carpeta en el directorio
    for folder_name in os.listdir(data_directory):
        folder_path = os.path.join(data_directory, folder_name)

        # Verifica si la ruta es un directorio
        if os.path.isdir(folder_path):
            # Lista los archivos en la carpeta
            files = os.listdir(folder_path)

            # Asegúrate de que haya al menos un archivo en la carpeta
            if files:
                # Añade la información al listado
                folder_info.append({
                    'nombre': folder_name,
                    'primer_elemento': files[0]
                })

    return render_template('personasregistradas.html', folder_info=folder_info)

def gen():
    dataPath = '../Data'
    imagePaths = os.listdir(dataPath)
    cap = cv2.VideoCapture(0)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('./modeloLBPHFace.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(roi)

            if result[1] < 90:
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y - 25), 2, 1.1, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)