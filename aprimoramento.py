import os
import shutil
import re

def obter_ultimo_numero_imagem(pasta_destino, id_cadastrado, nome):
    padrao = re.compile(rf'{id_cadastrado}_{nome}_(\d+).jpg')
    max_numero = 0
    for nome_arquivo in os.listdir(pasta_destino):
        correspondencia = padrao.match(nome_arquivo)
        if correspondencia:
            num = int(correspondencia.group(1))
            if num > max_numero:
                max_numero = num
    return max_numero

def verificar_id_cadastrado(info_file_path, id_cadastrado, nome):
    id_cadastrado = int(id_cadastrado)
    nome = nome.strip()  # Remover espaços em branco extras do nome
    
    id_cadastrado_existe = False

    if os.path.exists(info_file_path):
        with open(info_file_path, 'r') as file:
            for line in file:
                existing_id, existing_name = line.strip().split(',')
                existing_id = int(existing_id)
                if existing_id == id_cadastrado and existing_name == nome:
                    id_cadastrado_existe = True
                    break

    return id_cadastrado_existe

def adicionar_id_info(info_file_path, id_cadastrado, nome):
    with open(info_file_path, 'a') as info_file:
        info_file.write(f'{id_cadastrado},{nome}\n')
    print(f'ID {id_cadastrado} adicionado com sucesso ao arquivo info.txt.')

def renomear_e_mover_imagens(pasta_origem, pasta_destino, id_estranho, id_cadastrado, nome):
    info_file_path = 'info.txt'
    id_cadastrado_existe = verificar_id_cadastrado(info_file_path, id_cadastrado, nome)

    if not id_cadastrado_existe:
        adicionar_id_info(info_file_path, id_cadastrado, nome)

    ultimo_numero = obter_ultimo_numero_imagem(pasta_destino, id_cadastrado, nome)

    for nome_arquivo in os.listdir(pasta_origem):
        if nome_arquivo.startswith(id_estranho) and nome_arquivo.endswith('.jpg'):
            ultimo_numero += 1
            novo_nome_arquivo = f'{id_cadastrado}_{nome}_{ultimo_numero}.jpg'
            caminho_origem = os.path.join(pasta_origem, nome_arquivo)
            caminho_destino = os.path.join(pasta_destino, novo_nome_arquivo)

            # Mover o arquivo
            shutil.move(caminho_origem, caminho_destino)
            print(f'Movido: {nome_arquivo} para {novo_nome_arquivo}')

    print(f'Todas as imagens movidas e renomeadas para o ID cadastrado {id_cadastrado} ({nome}).')

def main():
    pasta_origem = 'estranhos'  # Pasta contendo as imagens desconhecidas
    pasta_destino = 'fotos'  # Pasta para mover as imagens renomeadas

    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    os.system('clear')
    id_estranho = input("Digite o ID do estranho para aprimorar: ")
    id_cadastrado = input("Digite o ID cadastrado da pessoa: ")
    nome = input("Digite o nome da pessoa: ")

    renomear_e_mover_imagens(pasta_origem, pasta_destino, id_estranho, id_cadastrado, nome)

if __name__ == '__main__':
    main()
