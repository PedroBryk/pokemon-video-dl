"""
utils.py - funcoes compartilhadas entre extracao, treino e deteccao.
REGRA DE OURO: a imagem deve ser preparada EXATAMENTE do mesmo jeito
no treino e na deteccao. Por isso esta funcao fica num arquivo so.
"""
import cv2
import numpy as np

IMG_SIZE = 224
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def letterbox(frame, size=IMG_SIZE):
    """Redimensiona mantendo a proporcao e completa com bordas pretas.
    Assim o objeto nao fica 'esticado' nem cortado."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas