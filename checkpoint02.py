import cv2
import numpy as np

video = cv2.VideoCapture('pedra-papel-tesoura.mp4')

ret, frame = video.read()
#ranges das areas que caracterizam pedra, papel e tesoura
range_pedra = (15800.0, 16000.0)
range_papel = (19300.0, 19600.0)
range_tesoura = (15200.0, 15600.0)

pontos_jogador1 = 0
pontos_jogador2 = 0
jogada_jogador1 = 0
jogada_jogador2 = 0

while True:

    ret, frame = video.read()
    frame = cv2.resize(frame, (1000, 600), 1)
    if not ret:
        break

    #aplica filtro cinza na imagem
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # aplica o detector de bordas de Canny
    edges = cv2.Canny(gray, 200, 400)
    # Converte o frame para o espaço de cores HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define a faixa de cor branca em HSV
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([255, 30, 255], dtype=np.uint8)
    # Cria uma máscara para a cor branca e pega o inverso dela, ou seja os objetos <> de branco no video
    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask_inv = cv2.bitwise_not(mask)
    
    # Encontra os contornos no video
    blurred = cv2.GaussianBlur(mask_inv, (5, 5), 0)
    contours, hierarchy = cv2.findContours(
        blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ordena os contornos em ordem decrescente de área para pegar os maiores
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    #separa os videos em esquerda e direita
    altura, largura, _ = frame.shape
    frame_esquerda = frame[:, :largura // 2, :]
    frame_direita = frame[:, largura // 2:, :]
    cv2.line(frame, (largura // 2, 0), (largura // 2, altura), (0, 0, 255), 2)

    
    # Verifica os dois contornos e faz um for para ir identificando as jogadas durante o video
    if len(contours) >= 2:
        for i in range(2):  
            contour = contours[i]
            area = cv2.contourArea(contour)

            cv2.putText(frame, f"Area : {area}", (int(contour.ravel()[0]), int(
                contour.ravel()[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


            # Desenha a área do contorno no video, juntamente com a jogada de cada jogador
            x, y, w, h = cv2.boundingRect(contour)
            if range_pedra[0] <= area <= range_pedra[1]:
                jogada = "pedra"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Pedra", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif range_papel[0] <= area <= range_papel[1]:
                jogada = "papel"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Papel", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            elif range_tesoura[0] <= area <= range_tesoura[1]:
                jogada = "tesoura"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Tesoura", (x, y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if contour[0][0][0] < largura // 2:
                jogada_jogador1 = jogada
            else:
                jogada_jogador2 = jogada
                
        # Faz as comparações das jogadas para definir os ganhadores de cada round, somando seus pontos
        if jogada_jogador1 == "pedra" and jogada_jogador2 == "tesoura":
            cv2.putText(frame_esquerda, "Jogador 1 ganhou", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 0), 2)
            pontos_jogador1 += 1
        elif jogada_jogador1 == "tesoura" and jogada_jogador2 == "papel":
            cv2.putText(frame_esquerda, "Jogador 1 ganhou", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 0), 2)
            pontos_jogador1 += 1
        elif jogada_jogador1 == "papel" and jogada_jogador2 == "pedra":
            cv2.putText(frame_esquerda, "Jogador 1 ganhou", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 0), 2)
            pontos_jogador1 += 1
        elif jogada_jogador1 == jogada_jogador2:
            cv2.putText(frame, "Empate", (390, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
            pass
        else:
            cv2.putText(frame_direita, "Jogador 2 ganhou", (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 200), 2)
            pontos_jogador2 += 1
        

    # Exibe o placar de cada jogador
    cv2.putText(frame_esquerda, f"Pontos Jogador 1: {pontos_jogador1}", (
        30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
    cv2.putText(frame_direita, f"Pontos Jogador 2: {pontos_jogador2}", (
        220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 0, 200), 2)
    
    # junta as telas esquerda e direita no video 
    frame = np.concatenate((frame_esquerda, frame_direita), axis=1)

    cv2.imshow('CHECKPOINT02', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()