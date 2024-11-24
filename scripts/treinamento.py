import cv2
import os
import numpy as np

# Cria o classificador LBPH
lbph = cv2.face.LBPHFaceRecognizer_create()  # type: ignore

def reorganizePhotos():
    '''
    Reorganiza as fotos na pasta 'fotos', garantindo que estejam enumeradas 
    sequencialmente começando do 1.
    '''
    output_dir = 'fotos'
    pathsImages = sorted([f for f in os.listdir(output_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    for i, filename in enumerate(pathsImages, start=1):
        # Extrai o nome e extensão do arquivo
        name, ext = os.path.splitext(filename)
        
        # Extrai o ID e o nome
        parts = name.split('_')
        if len(parts) < 2:
            print(f'Nome de arquivo inválido: {filename}')
            continue
        
        id = parts[0]
        nome = parts[1]
        
        # Renomeia o arquivo com o número correto na sequência
        new_filename = f'{id}_{nome}_{i}{ext}'
        os.rename(os.path.join(output_dir, filename), os.path.join(output_dir, new_filename))

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
    # Reorganiza as fotos antes de treinar
    reorganizePhotos()
    
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
