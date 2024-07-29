import cv2
import os
import numpy as np

# Cria o classificador LBPH
lbph = cv2.face.LBPHFaceRecognizer_create()  # type: ignore

def getImageWithId():
    '''
    Percorre o diretório 'fotos', lê todas as imagens com CV2 e organiza
    o conjunto de faces com seus respectivos IDs.
    '''
    pathsImages = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []

    for pathImage in pathsImages:
        if pathImage.endswith('.jpg') or pathImage.endswith('.png'):  # Verifica se é um arquivo de imagem
            imageFace = cv2.cvtColor(cv2.imread(pathImage), cv2.COLOR_BGR2GRAY)
            
            # Extrai o ID a partir do nome do arquivo
            id = os.path.basename(pathImage).split('_')[0]
            print(pathImage)

            # Verifica se o ID é numérico
            if id.isdigit():
                ids.append(int(id))
                faces.append(imageFace)
                
                # Mostra a imagem sendo treinada
                cv2.imshow("Treinando...", imageFace)
                cv2.waitKey(100)  # Espera 100ms para mostrar a imagem
            else:
                print(f"O ID do arquivo {pathImage} não é numérico.")

    cv2.destroyAllWindows()  # Fecha todas as janelas abertas
    return np.array(ids), faces

def trainRecognizer():
    '''
    Função para treinar o classificador LBPH e salvar o modelo treinado.
    '''
    ids, faces = getImageWithId()  # Obtém os IDs e faces das imagens

    # Verifica se há pelo menos uma imagem para treinar
    if len(ids) == 0 or len(faces) == 0:
        print("Nenhuma imagem encontrada para treinamento.")
        return

    # Treina o classificador LBPH com as faces e IDs obtidos
    lbph.train(faces, ids)

    # Salva o classificador treinado em um arquivo
    lbph.write('classifier/classificadorLBPH.yml')
    print('Treinamento concluído com sucesso!')

# Chamada da função para iniciar o treinamento
trainRecognizer()
