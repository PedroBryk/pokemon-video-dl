"""
Passo 3 - Reconhece objetos em um video (ou na webcam).
Uso:
  python detectar.py meu_video.mp4   -> gera saida_meu_video.mp4
  python detectar.py 0               -> usa a webcam (q = sair)
Cada frame e classificado sozinho: o modelo olha o frame, devolve as
probabilidades e o rotulo de maior probabilidade e escrito na tela.
A preparacao da imagem e a mesma do treino (letterbox do utils.py).
"""
import sys
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from utils import letterbox

CLASSE_VAZIA = "fundo"  # classe que significa "nenhum objeto"

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

n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if webcam:
        frame = cv2.flip(frame, 1)

    img = cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB).astype("float32")
    probs = model.predict(img[None], verbose=0)[0]
    i = int(np.argmax(probs))
    rotulo = class_names[i]

    # --- desenho ---
    texto = "nenhum objeto" if rotulo == CLASSE_VAZIA else rotulo
    cor = (0, 200, 0) if rotulo != CLASSE_VAZIA else (0, 200, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, f"{texto} ({probs[i]:.0%})", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
    for j, c in enumerate(class_names):
        y = 55 + j * 22
        cv2.rectangle(frame, (10, y), (10 + int(probs[j] * 150), y + 14), (0, 200, 255), -1)
        cv2.putText(frame, c, (170, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    cv2.putText(frame, f"t = {n / fps:5.1f}s", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if writer is not None:
        writer.write(frame)
    if webcam or os.environ.get("MOSTRAR", "1") == "1":
        cv2.imshow("Reconhecimento com CNN", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    n += 1

cap.release()
if writer is not None:
    writer.release()
    print(f"Video anotado salvo em: {saida}")
cv2.destroyAllWindows()