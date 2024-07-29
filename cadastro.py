import cv2
import os
import numpy as np
import time
from colorama import init, Fore

def Cadastro():
    # Caminho Haarcascade
    face_cascade_path = 'cascade/haarcascade_frontalface_default.xml'
    eye_cascade_path = 'cascade/haarcascade-eye.xml'

    # Classificador baseado nos Haarcascade
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    video_capture = cv2.VideoCapture(0)

    increment = 1
    numMostras = 70
    width, height = 220, 220

    while True:
        os.system('clear')
        id = input('Digite seu identificador: ')
        nome = input('Digite seu nome: ')

        # Cria diretório para salvar as informações e as fotos, se não existir
        output_dir = 'fotos'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Verificar se o ID e o nome já existem no arquivo
        info_file_path = 'info.txt'
        existing_ids = {}
        if os.path.exists(info_file_path):
            with open(info_file_path, 'r') as file:
                for line in file:
                    existing_id, existing_name = line.strip().split(',')
                    existing_ids[int(existing_id)] = existing_name

        if int(id) in existing_ids:
            if existing_ids[int(id)] == nome:
                print(f'ID {id} já cadastrado com o nome {nome}. Atualizando as fotos...')
            else:
                print(f'{Fore.RED}Erro:{Fore.RESET} ID {id} já cadastrado com o nome {existing_ids[int(id)]}. Use um ID diferente.')
                time.sleep(2)
                continue
        else:
            # Salva as informações de identificação no arquivo se for um novo cadastro
            with open(info_file_path, 'a') as info_file:
                info_file.write(f'{id},{nome}\n')

        print('Capturando as faces...')
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Falha na captura de imagem")
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Realizando detecção de face
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.5,
                minSize=(35, 35),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            for (x, y, w, h) in faces:
                # Desenhando retângulo na face detectada
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

                # Realizando detecção dos olhos da face
                face_region = frame[y:y + h, x:x + w]
                face_region_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                eyes = eye_cascade.detectMultiScale(face_region_gray)

                if len(eyes) == 2:
                    # Captura a imagem automaticamente
                    face_off = cv2.resize(gray[y:y + h, x:x + w], (width, height))
                    photo_name = f'{id}_{nome}_{increment}.jpg'
                    photo_path = os.path.join(output_dir, photo_name)
                    cv2.imwrite(photo_path, face_off)

                    print(f'[Foto {increment} capturada com sucesso] - Qualidade da luz: {np.average(gray):.2f}')
                    increment += 1

                    # Adiciona um pequeno atraso para evitar capturas muito rápidas
                    time.sleep(0.5)

            cv2.imshow('Cadastro', frame)
            
            if increment > numMostras:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Verificar se a janela foi fechada pelo botão de fechar (X)
            if cv2.getWindowProperty('Cadastro', cv2.WND_PROP_VISIBLE) < 1:
                break

        print('Fotos capturadas com sucesso :)')
        video_capture.release()
        cv2.destroyAllWindows()
