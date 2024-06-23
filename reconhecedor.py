from codecs import ignore_errors
from shutil import ignore_patterns
from tokenize import Ignore
import cv2
import numpy as np
from datetime import datetime
from collections import deque, Counter
from colorama import init, Fore

# Inicializar colorama
init()

# Função para carregar IDs e nomes
def load_id_names(filename):
    id_names = {}
    with open(filename, 'r') as file:
        for line in file:
            id, name = line.strip().split(',')
            id_names[int(id)] = name
    return id_names

# Caminho dos modelosa
model_path = 'modelos/'

# Caminho haarcascade
detectorFace = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

# Instanciando LBPH Faces Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create() #type: ignore
recognizer.read("classifier/classificadorLBPH.yml")

# Modelos para idade e gênero
faceProto = model_path + "opencv_face_detector.pbtxt"
faceModel = model_path + "opencv_face_detector_uint8.pb"
ageProto = model_path + "age_deploy.prototxt"
ageModel = model_path + "age_net.caffemodel"
genderProto = model_path + "gender_deploy.prototxt"
genderModel = model_path + "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2 anos)', '(4-6 anos)', '(8-12 anos)', '(15-20 anos)', '(25-30 anos)',
           '(30-37 anos)', '(38-42 anos)', '(48-53 anos)', '(60-70 anos)']
genderList = ['Homem', 'Mulher']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

height, width = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

# Limiar de confiança
limiar_confianca = 60

# Carregar IDs e nomes
id_names = load_id_names('info.txt')

# Dicionário para rastrear IDs de faces
face_ids = {}

# Dicionário para rastrear previsões de gênero e idade
face_info = {}

# Variáveis para armazenar o último log registrado
last_log = {}


# Próximo ID a ser atribuído
next_id = 1
next_unknown_id = 1

# Variável de filtro
frames_to_confirm = 500  # Ajuste este valor para mais precisão ou reatividade

def weighted_moving_average(values, alpha=0.6):
    """
    Calcula a média móvel ponderada com fator de suavização alpha.
    """
    avg = 0
    for i in range(len(values)):
        avg = alpha * values[i] + (1 - alpha) * avg
    return avg

while True:
    conectado, imagem = camera.read()
    imageGray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Detecção da face baseado no haarcascade
    faceDetect = detectorFace.detectMultiScale(
        imageGray,
        scaleFactor=1.2,   # Ajuste o fator de escala conforme necessário
        minNeighbors=7,     # Ajuste o número mínimo de vizinhos conforme necessário
        minSize=(100, 100),   # Ajuste o tamanho mínimo do objeto conforme necessário
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Atualizar lista de IDs ativos
    active_ids = list(face_ids.keys())

    # Remover IDs inativos
    for face_id in active_ids:
        if face_id not in [face_id for face_id, (_, face_position) in face_ids.items()]:
            del face_ids[face_id]

    # Associar IDs a novas faces detectadas
    for (x, y, w, h) in faceDetect:
        face_image = cv2.resize(imageGray[y:y+h, x:x+w], (width, height))
        
        # Fazendo comparação da imagem detectada
        id, confianca = recognizer.predict(face_image)
        
        # Convertendo confiança para porcentagem
        confianca_pct = 100 - confianca

        # Verificar se a face é conhecida baseada na confiança
        if confianca <= limiar_confianca:
            name = id_names.get(id, 'Estranho')
            if id == -1:
                id = 0  # Define o ID do desconhecido como 0 na saída do log
                confianca = 0
            log_color = Fore.GREEN  # Verde para conhecido
            rectangle_color = (0, 255, 0)  # Verde para conhecido
        else:
            # Verificar se já há um ID atribuído para este rosto
            found_match = False
            for face_id, (_, face_position) in face_ids.items():
                (prev_x, prev_y, prev_w, prev_h) = face_position
                overlap_x = max(0, min(x + w, prev_x + prev_w) - max(x, prev_x))
                overlap_y = max(0, min(y + h, prev_y + prev_h) - max(y, prev_y))
                overlap_area = overlap_x * overlap_y
                if overlap_area > 0:
                    face_ids[face_id] = ((x, y, w, h), face_position)
                    id = face_id
                    found_match = True
                    break

            if not found_match:
                unknown_id = f"00{next_unknown_id}"
                face_ids[unknown_id] = ((x, y, w, h), (x, y, w, h))
                id = unknown_id
                confianca = 0
                next_unknown_id += 1

            name = 'Estranho'
            log_color = Fore.RED  # Vermelho para desconhecido
            rectangle_color = (0, 0, 255)  # Vermelho para desconhecido

        # Adicionar detecção de idade e gênero
        face = imagem[max(0, y - 20):min(y + h + 20, imagem.shape[0] - 1), max(0, x - 20):min(x + w + 20, imagem.shape[1] - 1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Predição de gênero
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        # Predição de idade
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # Atualizar informações de gênero e idade
        if id not in face_info:
            face_info[id] = {'genders': deque(maxlen=frames_to_confirm), 'ages': deque(maxlen=frames_to_confirm)}
        
        face_info[id]['genders'].append(gender)
        face_info[id]['ages'].append(age)

        # Calcular a média móvel ponderada
        most_common_gender = weighted_moving_average([1 if g == 'Homem' else 0 for g in face_info[id]['genders']])
        most_common_age = Counter(face_info[id]['ages']).most_common(1)[0][0]

        # Determinar gênero com base na média ponderada
        most_common_gender = 'Homem' if most_common_gender >= 0.5 else 'Mulher'

        # Formatar a data e hora sem os segundos
        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M')

        # Preparar a linha de log
        log_line = {
            'id': id,
            'time': current_time_str,
            'nome': name,
            'gênero': most_common_gender,
            'idade': most_common_age
        }

        # Exibir a linha de log no terminal apenas se houver alterações
        if id not in last_log or {k: log_line[k] for k in log_line}:
            last_log[id] = log_line
            if log_color == Fore.GREEN:
                print(f"id: {Fore.GREEN}{id}{Fore.RESET}, time: {current_time_str}, nome: {Fore.GREEN}{name}{Fore.RESET}, confiança: {int(confianca_pct)}%, gênero: {most_common_gender}, idade: {most_common_age}")
            elif log_color == Fore.RED:
                print(f"id: {Fore.RED}{id}{Fore.RESET}, time: {current_time_str}, nome: {Fore.RED}{name}{Fore.RESET}, confiança: {int(confianca_pct)}%, gênero: {most_common_gender}, idade: {most_common_age}")

        # Desenhar retângulo em volta da face
        cv2.rectangle(imagem, (x, y), (x + w, y + h), rectangle_color, 2)  # Retângulo em vermelho

        # Definir texto a ser exibido para ID, nome e porcentagem de confiança
        display_text = f"ID: {id}, {name}, {int(confianca_pct)}%"
        (text_width,         text_height), _ = cv2.getTextSize(display_text, font, fontScale=1, thickness=1)
        text_x = x
        text_y = y - 10
        text_bg_width = text_width + 10
        text_bg_height = text_height + 10
        text_bg_rect = ((text_x, text_y), (text_x + text_bg_width, text_y - text_bg_height))

        # Desenhar fundo do texto (retângulo preto)
        cv2.rectangle(imagem, text_bg_rect[0], text_bg_rect[1], (0, 0, 0), cv2.FILLED)

        # Desenhar texto sobre a imagem
        cv2.putText(imagem, display_text, (text_x + 5, text_y - 5), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Definir posição e tamanho do fundo do texto de gênero e idade
        gender_age_text = f'{most_common_gender}, {most_common_age}'
        (gender_age_text_width, gender_age_text_height), _ = cv2.getTextSize(gender_age_text, font, fontScale=1, thickness=1)
        gender_age_text_x = x
        gender_age_text_y = y + h + 24
        gender_age_text_bg_width = gender_age_text_width + 10
        gender_age_text_bg_height = gender_age_text_height + 10
        gender_age_text_bg_rect = ((gender_age_text_x, gender_age_text_y), (gender_age_text_x + gender_age_text_bg_width, gender_age_text_y - gender_age_text_bg_height))

        # Desenhar fundo do texto de gênero e idade (retângulo preto)
        cv2.rectangle(imagem, gender_age_text_bg_rect[0], gender_age_text_bg_rect[1], (0, 0, 0), cv2.FILLED)

        # Desenhar texto de gênero e idade sobre a imagem
        cv2.putText(imagem, gender_age_text, (gender_age_text_x + 5, gender_age_text_y - 5), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Mostrando frame
    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

