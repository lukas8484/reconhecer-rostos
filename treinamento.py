import cv2
import os
import numpy as np

# Usando 3 algoritmos de face detect
eigenface = cv2.face.EigenFaceRecognizer_create() # type: ignore
fisherface = cv2.face.FisherFaceRecognizer_create() # type: ignore
lbph = cv2.face.LBPHFaceRecognizer_create() # type: ignore

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

            # Verifica se o ID é numérico
            if id.isdigit():
                ids.append(int(id))
                faces.append(imageFace)
                
                # Mostra a imagem sendo treinada
                cv2.imshow("Treinando...", imageFace)
                cv2.waitKey(500)  # Espera 500ms para mostrar a imagem
            else:
                print(f"O ID do arquivo {pathImage} não é numérico.")

    cv2.destroyAllWindows()  # Fecha todas as janelas abertas
    return np.array(ids), faces

ids, faces = getImageWithId()

# Gerando classifier do treinamento
print("Treinando....")
eigenface.train(faces, ids)
eigenface.write('classifier/classificadorEigen.yml')

# fisherface.train(faces, ids)
# fisherface.write('classifier/classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classifier/classificadorLBPH.yml')
print('Treinamento concluído com sucesso!')