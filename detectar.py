"""
Passo 3 - Reconhece Pokemon em um video (ou na webcam).
Uso:
  python detectar.py meu_video.mp4   -> gera saida_meu_video.mp4
  python detectar.py 0               -> usa a webcam (q = sair)

O modelo classifica cada frame, mas o rotulo mostrado NAO e mais o do frame
sozinho. Duas melhorias foram implementadas:
  MELHORIA 1: suavizacao (media das probabilidades dos ultimos N frames) e
              limiar de confianca (abaixo dele, mostra "?").
  MELHORIA 2: linha do tempo impressa no final (trechos de cada Pokemon).
A preparacao da imagem e a mesma do treino (letterbox do utils.py).

Para testar outros valores sem editar o codigo (Windows):
  set JANELA=15
  set LIMIAR=0.7
  python detectar.py meu_video.mp4
"""
import sys
import os
import json
from collections import deque
import cv2
import numpy as np
import tensorflow as tf
from utils import letterbox

CLASSE_VAZIA = "fundo"  # classe que significa "nenhum objeto"

# --- MELHORIA 1: parametros da suavizacao e do limiar ---
JANELA = int(os.environ.get("JANELA", 8))       # quantos frames entram na media
LIMIAR = float(os.environ.get("LIMIAR", 0.60))  # confianca minima para mostrar um nome
# --- MELHORIA 2: parametros da linha do tempo ---
MIN_DURACAO = 0.5  # segundos: trechos mais curtos que isso sao ignorados
MAX_LACUNA = 1.0   # segundos: junta trechos do mesmo Pokemon separados por pausa curta (0 = desliga)

model = tf.keras.models.load_model("modelo.keras")
class_names = json.load(open("class_names.json"))

fonte = sys.argv[1] if len(sys.argv) > 1 else "0"
webcam = fonte.isdigit()
cap = cv2.VideoCapture(int(fonte) if webcam else fonte)
if not cap.isOpened():
    raise SystemExit(f"Nao consegui abrir: {fonte}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30

writer = None
if not webcam:
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saida = f"saida_{os.path.splitext(os.path.basename(fonte))[0]}.mp4"
    writer = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# MELHORIA 1: o deque guarda so as probabilidades dos ultimos JANELA frames
historico = deque(maxlen=JANELA)
# MELHORIA 2: guarda o rotulo decidido em cada frame, para montar a linha do tempo
rotulos = []

n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if webcam:
        frame = cv2.flip(frame, 1)

    img = cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB).astype("float32")
    probs = model.predict(img[None], verbose=0)[0]

    # --- MELHORIA 1: suavizacao + limiar ---
    historico.append(probs)
    media = np.mean(historico, axis=0)   # media das probabilidades dos ultimos frames
    i = int(np.argmax(media))
    if media[i] < LIMIAR:                # confianca baixa: nao chuta um nome
        rotulo = "?"
    else:
        rotulo = class_names[i]
    rotulos.append(rotulo)               # MELHORIA 2: guarda o rotulo deste frame

    # --- desenho (usa a media suavizada, nao a probabilidade do frame sozinho) ---
    if rotulo == "?":
        texto, cor = "?", (0, 0, 255)
    elif rotulo == CLASSE_VAZIA:
        texto, cor = "nenhum objeto", (0, 200, 255)
    else:
        texto, cor = rotulo, (0, 200, 0)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, f"{texto} ({media[i]:.0%})", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
    for j, c in enumerate(class_names):
        y = 55 + j * 22
        cv2.rectangle(frame, (10, y), (10 + int(media[j] * 150), y + 14), (0, 200, 255), -1)
        cv2.putText(frame, c, (170, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    cv2.putText(frame, f"t = {n / fps:5.1f}s", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"janela={JANELA}  limiar={LIMIAR:.0%}", (170, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if writer is not None:
        writer.write(frame)
    if webcam or os.environ.get("MOSTRAR", "1") == "1":
        cv2.imshow("Reconhecimento de Pokemon", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    n += 1

cap.release()
if writer is not None:
    writer.release()
    print(f"Video anotado salvo em: {saida}")
cv2.destroyAllWindows()


# ---- MELHORIA 2: linha do tempo ----------------------------------------------
def linha_do_tempo(rotulos, fps, min_dur=MIN_DURACAO, max_lacuna=MAX_LACUNA):
    """Imprime em que trechos cada Pokemon apareceu e a lista dos encontrados."""
    # 1) agrupa frames seguidos com o mesmo rotulo: [rotulo, inicio_s, fim_s]
    #    (o tempo de um frame e indice / fps)
    trechos = []
    ini = 0
    for k in range(1, len(rotulos) + 1):
        if k == len(rotulos) or rotulos[k] != rotulos[ini]:
            trechos.append([rotulos[ini], ini / fps, k / fps])
            ini = k
    # 2) descarta os trechos de "?", de fundo e os muito curtos (< min_dur)
    validos = [t for t in trechos
               if t[0] not in ("?", CLASSE_VAZIA) and (t[2] - t[1]) >= min_dur]
    # 3) junta trechos do mesmo Pokemon separados por uma pausa curta
    #    (evita que um "?" de meio segundo no meio parta o trecho em dois)
    fundidos = []
    for t in validos:
        if fundidos and fundidos[-1][0] == t[0] and t[1] - fundidos[-1][2] <= max_lacuna:
            fundidos[-1][2] = t[2]
        else:
            fundidos.append(t)

    print("\nLinha do tempo (Pokemon identificados):")
    for nome, t0, t1 in fundidos:
        print(f"{t0:.1f}s a {t1:.1f}s -> {nome}")
    achados = sorted({t[0] for t in fundidos})
    print(f"Pokemon encontrados no video: {achados}")


linha_do_tempo(rotulos, fps)