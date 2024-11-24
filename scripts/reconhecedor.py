import cv2
from datetime import datetime
from collections import deque, Counter
from colorama import init, Fore
import os

def find_camera(max_index=16):
    print("Procurando câmera conectada...")
    index = 0
    while index <= max_index:
        cap = cv2.VideoCapture(index)
        print(f"Tentativa no índice {index}")
        if cap.isOpened():
            cap.release()
            return index
        cap.release()
        index += 1
    return -1  # Retorna -1 caso nenhuma câmera seja encontrada até o índice máximo

# Use a função find_camera() para encontrar o índice da câmera conectada
camera_index = find_camera()
if camera_index != -1:
    print(f"Câmera conectada encontrada no índice {camera_index}")
else:
    print("Nenhuma câmera encontrada até o índice 8.")

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
upper_cascade = cv2.CascadeClassifier('cascade/haarcascade_upperbody.xml')

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
camera = cv2.VideoCapture(camera_index)

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

# Dicionário para rastrear IDs de corpos
body_ids = {}
next_body_id = 1

# Dicionário para rastrear IDs e nomes dos corpos
body_names = {}

# Dicionário para rastrear IDs de corpos e associar com rostos
body_face_mapping = {}

# Variável de filtro
frames_to_confirm = 500  # Ajuste este valor para mais precisão ou reatividade

# Criar pasta para armazenar imagens de desconhecidos
if not os.path.exists('estranhos'):
    os.makedirs('estranhos')

# Função para calcular a média móvel ponderada
def weighted_moving_average(values, alpha=0.6):
    avg = 0
    for i in range(len(values)):
        avg = alpha * values[i] + (1 - alpha) * avg
    return avg

# Contador de imagens para cada desconhecido
unknown_counters = {}

# Remover IDs inativos (Faces e Corpos)
def remove_inactive_ids(active_ids, id_dict, info_dict=None, mapping_dict=None):
    for id in list(id_dict.keys()):
        if id not in active_ids:
            del id_dict[id]
            if info_dict is not None and id in info_dict:
                del info_dict[id]
            if mapping_dict is not None and id in mapping_dict:
                del mapping_dict[id]

# Lista de IDs ativos
active_face_ids = []
active_body_ids = []

os.system('clear')
while True:

    conectado, imagem = camera.read()
    imageGray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Detecção da face baseado no haarcascade
    faceDetect = detectorFace.detectMultiScale(
        imageGray,
        scaleFactor=1.2,   # Ajuste o fator de escala conforme necessário
        minNeighbors=7,     # Ajuste o número mínimo de vizinhos conforme necessário
        minSize=(70, 70),   # Ajuste o tamanho mínimo do objeto conforme necessário
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Ajustar parâmetros para detectar corpos maiores
    upper = upper_cascade.detectMultiScale(
        imageGray,
        scaleFactor=1.01,
        minNeighbors=7,
        minSize=(160, 160)
    )
 
    # Atualizar lista de IDs ativos (Faces)
    for (x, y, w, h) in faceDetect:
        face_image = cv2.resize(imageGray[y:y+h, x:x+w], (width, height))
        id, confianca = recognizer.predict(face_image)
        active_face_ids.append(id)
        face_ids[id] = (x, y, w, h)

    # Desenhar corpos
    for (x, y, w, h) in upper:

        found_body_match = False
        for body_id, (prev_x, prev_y, prev_w, prev_h) in body_ids.items():
            overlap_x = max(0, min(x + w, prev_x + prev_w) - max(x, prev_x))
            overlap_y = max(0, min(y + h, prev_y + prev_h) - max(y, prev_y))
            overlap_area = overlap_x * overlap_y

            if not overlap_area > 0:
                # O corpo ainda está na cena
                found_body_match = False
                body_names[body_id] = 'Estranho'  # Inicializar o nome como desconhecido
                body_face_mapping[body_id] = None  # Associar ID do corpo ao rosto (inicialmente None)
                break
            

            if overlap_area > 0:
                # O corpo ainda está na cena
                body_ids[body_id] = (x, y, w, h)
                found_body_match = True
                break

        if not found_body_match:
            body_id = next_body_id
            body_ids[body_id] = (x, y, w, h)
            next_body_id += 1
            if body_id not in body_names:
                body_names[body_id] = 'Estranho'  # Inicializar o nome como desconhecido
                body_face_mapping[body_id] = None  # Associar ID do corpo ao rosto (inicialmente None)

        # Verificar rosto dentro do retângulo do corpo
        face_detected = False
        for (fx, fy, fw, fh) in faceDetect:
            if fx >= x and fy >= y and fx + fw <= x + w and fy + fh <= y + h:
                # O rosto está dentro do retângulo do corpo
                face_image = cv2.resize(imageGray[fy:fy+fh, fx:fx+fw], (width, height))
                id, confianca = recognizer.predict(face_image)
                if confianca <= limiar_confianca:
                    person_name = id_names.get(id, 'Estranho')
                    if body_face_mapping[body_id] is None or body_face_mapping[body_id] == id:
                        body_names[body_id] = person_name  # Associar o nome ao corpo
                        body_face_mapping[body_id] = id  # Associar o rosto ao ID do corpo
                        face_detected = True
                    break

        # Se nenhum rosto foi detectado dentro do retângulo
        if not face_detected:
            if body_id not in body_face_mapping or body_face_mapping[body_id] is None:
                # Apenas resetar o nome se o corpo não tiver um nome associado
                body_names[body_id] = 'Estranho'
                body_face_mapping[body_id] = None

        # Desenhar retângulo e texto
        cv2.rectangle(imagem, (x, y), (x + w, y + h + 200), (255, 255, 255), 2)
        body_text = f"{body_names.get(body_id, 'Estranho')}"
        cv2.putText(imagem, body_text, (x, y - 10), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Remover IDs inativos (Corpos)
    active_body_ids = []
    # Atualizar lista de IDs ativos (Corpos)
    for (x, y, w, h) in upper:
        found_body_match = False
        for body_id, (prev_x, prev_y, prev_w, prev_h) in body_ids.items():
            overlap_x = max(0, min(x + w, prev_x + prev_w) - max(x, prev_x))
            overlap_y = max(0, min(y + h, prev_y + prev_h) - max(y, prev_y))
            overlap_area = overlap_x * overlap_y
            if overlap_area > 0:
                body_ids[body_id] = (x, y, w, h)
                found_body_match = True
                active_body_ids.append(body_id)
                break

        if not found_body_match:
            body_id = next_body_id
            body_ids[body_id] = (x, y, w, h)
            next_body_id += 1
            active_body_ids.append(body_id)

    # Remover IDs inativos de faces e corpos
    remove_inactive_ids(active_face_ids, face_ids, face_info)
    remove_inactive_ids(active_body_ids, body_ids, body_names, body_face_mapping)

    # Verificar quais IDs não estão ativos
    for face_id in list(face_ids.keys()):
        if face_id not in active_face_ids:
            # Remover face inativa
            del face_ids[face_id]
            if face_id in face_info:
                del face_info[face_id]


    # Remove faces que não estão mais ativas
    for face_id in list(face_ids.keys()):
        if face_id not in active_face_ids:
            del face_ids[face_id]
            face_info.pop(face_id, None)



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
            for face_id, face_data in face_ids.items():
                if len(face_data) == 4:
                    x, y, w, h = face_data
                    face_position = (x, y, w, h)
                    (prev_x, prev_y, prev_w, prev_h) = face_position
                    overlap_x = max(0, min(x + w, prev_x + prev_w) - max(x, prev_x))
                    overlap_y = max(0, min(y + h, prev_y + prev_h) - max(y, prev_y))
                    overlap_area = overlap_x * overlap_y
                    if overlap_area > 0:
                        face_ids[face_id] = ((x, y, w, h), face_position)
                        id = face_id
                        found_match = True
                        break
                    else:
                        print(f"Dados incompletos para face_id {face_id}: {face_data}")

            if not found_match:
                #Capturar estranho apenas quando a tecla K é pressionada
                unknown_id = f"00{next_unknown_id}"
                face_ids[unknown_id] = ((x, y, w, h), (x, y, w, h))
                id = unknown_id
                confianca = 0
                next_unknown_id += 1
            else:
           
                confianca = 0

            name = 'Estranho'
            log_color = Fore.RED  # Vermelho para desconhecido
            rectangle_color = (0, 0, 255)  # Vermelho para desconhecido
        
        key = cv2.waitKey(1)

        if key == 13:  # Capturar estranhos apenas quando a tecla K é pressionada
            # Salvar imagem do estranho
            if unknown_id not in unknown_counters:
                unknown_counters[unknown_id] = 1
            else:
                unknown_counters[unknown_id] += 1

            # Recortar a face e converter para preto e branco
            face_crop = imagem[y:y+h, x:x+w]
            face_crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

            # Definir o nome do arquivo
            filename = f'estranhos/{unknown_id}_{unknown_counters[unknown_id]}.jpg'

            # Salvar a imagem em preto e branco
            cv2.imwrite(filename, face_crop_gray)
            print(f'Imagem do estranho salva: {filename}')


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
        (text_width, text_height), _ = cv2.getTextSize(display_text, font, fontScale=1, thickness=1)
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
    cv2.imshow("Reconhecedor", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

    # Verificar se a janela foi fechada pelo botão de fechar (X)
    if cv2.getWindowProperty('Reconhecedor', cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()