"""
Passo 2 - Treino com Transfer Learning (MobileNetV2) + FINE-TUNING.
Uso: python train.py

FASE 1: a base (MobileNetV2, pre-treinada na ImageNet) fica congelada e so a
"cabeca" nova aprende, igual a atividade anterior.
FASE 2 (MELHORIA 3): descongela as ultimas camadas da base e continua o treino
com uma taxa de aprendizado bem menor (fine-tuning).
A validacao e separada POR VIDEO nas duas fases.
"""
import os
import json
import random
from collections import defaultdict
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "data"
IMG_SIZE = 224
BATCH = 16
EPOCHS = 8
VAL_FRACTION = 0.2  # fracao de VIDEOS (nao de fotos) para validacao
SEED = 42
# --- MELHORIA 3 (fine-tuning): parametros da segunda fase ---
EPOCHS_FT = 8       # maximo de epocas da fase 2
N_DESCONGELAR = 30  # quantas camadas finais da MobileNetV2 voltam a treinar
LR_FT = 1e-5        # taxa de aprendizado bem menor que a da fase 1 (1e-3)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

class_names = sorted(d for d in os.listdir(DATA_DIR)
                     if os.path.isdir(os.path.join(DATA_DIR, d)))
print("Classes:", class_names)
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

# ---- Split POR VIDEO ------------------------------------------------------
# Frames do mesmo video sao quase iguais. Se um video tiver frames no treino e
# na validacao, a acuracia fica enganosamente alta. Por isso cada video inteiro
# vai OU para treino OU para validacao. O nome do video vem antes de "__".
train_paths, train_y, val_paths, val_y = [], [], [], []
for idx, cls in enumerate(class_names):
    por_video = defaultdict(list)
    for f in sorted(os.listdir(os.path.join(DATA_DIR, cls))):
        if f.lower().endswith(".jpg"):
            por_video[f.split("__")[0]].append(os.path.join(DATA_DIR, cls, f))
    videos = sorted(por_video)
    if len(videos) < 2:
        raise SystemExit(f"'{cls}' tem {len(videos)} video(s). Grave pelo menos 3 por classe.")
    random.Random(SEED).shuffle(videos)
    n_val = max(1, round(len(videos) * VAL_FRACTION))
    val_videos = set(videos[:n_val])
    for v in videos:
        destino = (val_paths, val_y) if v in val_videos else (train_paths, train_y)
        destino[0].extend(por_video[v]); destino[1].extend([idx] * len(por_video[v]))
    n_tr = sum(len(por_video[v]) for v in videos if v not in val_videos)
    n_va = sum(len(por_video[v]) for v in val_videos)
    print(f"  {cls}: {n_tr} treino / {n_va} validacao | videos de validacao: {sorted(val_videos)}")


def load(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, label


def make_ds(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train_paths, train_y, True)
val_ds = make_ds(val_paths, val_y, False)

# Data augmentation (so atua no treino). O flip horizontal simula o objeto
# virado para o outro lado; a rotacao e o zoom simulam a camera se mexendo.
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.20),
    tf.keras.layers.RandomTranslation(0.10, 0.10),
    tf.keras.layers.RandomBrightness(0.25),
    tf.keras.layers.RandomContrast(0.25),
])

base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False,
    weights="imagenet")
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)


def avaliar(titulo):
    """Imprime a loss, a matriz de confusao e a precisao/recall na validacao.
    Chamada apos cada fase para podermos comparar antes e depois."""
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    y_true = np.array(val_y)
    y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)
    k = len(class_names)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    print(f"\n===== {titulo} =====")
    print("Matriz de confusao (linhas = real, colunas = previsto):")
    print(" " * 11 + " ".join(f"{c[:8]:>8}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"{c[:10]:>10} " + " ".join(f"{v:>8}" for v in cm[i]))
    print("\nClasse        precisao   recall")
    for i, c in enumerate(class_names):
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        rec = cm[i, i] / max(cm[i, :].sum(), 1)
        print(f"{c[:12]:<12} {prec:>9.2%} {rec:>8.2%}")
    print(f"\nAcuracia de validacao: {val_acc:.2%} | loss de validacao: {val_loss:.4f}")


# ==== FASE 1: treina so a cabeca (base congelada) ============================
print("\n##### FASE 1: treino da cabeca (base congelada) #####")
parar = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                         restore_best_weights=True)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

hist1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[parar])
model.save("modelo_base.keras")  # guarda a versao da fase 1 para comparar depois
avaliar("FASE 1 - base congelada")

# ==== FASE 2 (MELHORIA 3): FINE-TUNING ======================================
# Descongela so as ultimas N_DESCONGELAR camadas da MobileNetV2 e continua o
# treino com taxa de aprendizado bem menor. A cabeca ja foi treinada na fase 1;
# se a base fosse descongelada com a cabeca aleatoria, os gradientes grandes
# destruiriam os filtros pre-treinados.
print(f"\n##### FASE 2: fine-tuning das ultimas {N_DESCONGELAR} camadas (lr={LR_FT}) #####")
base.trainable = True
for layer in base.layers[:-N_DESCONGELAR]:
    layer.trainable = False

# OBRIGATORIO recompilar depois de mudar o que e treinavel
model.compile(optimizer=tf.keras.optimizers.Adam(LR_FT),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
n_treinaveis = int(sum(np.prod(w.shape) for w in model.trainable_weights))
print(f"Parametros treinaveis na fase 2: {n_treinaveis:,}")

parar_ft = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                            restore_best_weights=True)
n1 = len(hist1.history["loss"])  # epocas realmente executadas na fase 1
hist2 = model.fit(train_ds, validation_data=val_ds,
                  epochs=n1 + EPOCHS_FT, initial_epoch=n1, callbacks=[parar_ft])

model.save("modelo.keras")  # modelo final (o detectar.py usa este arquivo)

# ---- Curvas de treino vs validacao (fase 1 + fase 2) ----------------------
# MELHORIA 3: o grafico junta os dois historicos e marca onde comeca o fine-tuning.
plt.figure(figsize=(10, 4))
for i, (m, t) in enumerate([("accuracy", "Acuracia"), ("loss", "Loss")]):
    plt.subplot(1, 2, i + 1)
    plt.plot(hist1.history[m] + hist2.history[m], label="treino")
    plt.plot(hist1.history["val_" + m] + hist2.history["val_" + m], label="validacao")
    plt.axvline(n1 - 0.5, color="red", linestyle="--", label="inicio do fine-tuning")
    plt.title(t); plt.xlabel("epoca"); plt.legend()
plt.tight_layout(); plt.savefig("curvas.png")

avaliar("FASE 2 - apos fine-tuning (modelo final)")

# ---- PDF do dataset (entrega obrigatoria) ---------------------------------
from PIL import Image, ImageDraw, ImageFont


def gerar_pdf_dataset(data_dir="data", saida="dataset_imagens.pdf", cols=10,
                      qualidade=70, max_por_classe=300):
    """Uma secao por classe. Se houver muitas fotos, pega uma amostra espacada
    (max_por_classe) para o PDF nao ficar gigante. A legenda mostra quantos
    videos geraram as fotos daquela classe."""
    pag_w, pag_h, margem, topo = 1240, 1754, 40, 110
    cell = (pag_w - 2 * margem) // cols
    rows = (pag_h - topo - margem) // cell
    por_pagina = cols * rows
    try:
        fonte = ImageFont.load_default(size=28)
    except TypeError:
        fonte = ImageFont.load_default()
    paginas = []
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    for classe in classes:
        pasta = os.path.join(data_dir, classe)
        fotos = sorted(f for f in os.listdir(pasta) if f.lower().endswith(".jpg"))
        n_videos = len({f.split("__")[0] for f in fotos})
        total = len(fotos)
        if total > max_por_classe:
            idx = np.linspace(0, total - 1, max_por_classe).astype(int)
            fotos = [fotos[i] for i in idx]
        n_pag = max(1, -(-len(fotos) // por_pagina))
        for p in range(n_pag):
            pagina = Image.new("RGB", (pag_w, pag_h), "white")
            draw = ImageDraw.Draw(pagina)
            draw.text((margem, 40),
                      f"Classe: {classe} | {total} imagens de {n_videos} videos | pagina {p + 1}/{n_pag}",
                      fill="black", font=fonte)
            for i, nome in enumerate(fotos[p * por_pagina:(p + 1) * por_pagina]):
                img = Image.open(os.path.join(pasta, nome)).convert("RGB")
                img.thumbnail((cell - 6, cell - 6))
                pagina.paste(img, (margem + (i % cols) * cell + 3,
                                   topo + (i // cols) * cell + 3))
            paginas.append(pagina)
    paginas[0].save(saida, save_all=True, append_images=paginas[1:],
                    quality=qualidade, resolution=150)
    print(f"PDF salvo: {saida} ({len(paginas)} paginas)")


gerar_pdf_dataset(DATA_DIR)
print("\nPronto! Modelo final em modelo.keras (versao da fase 1 em modelo_base.keras).")
print("Agora rode: python detectar.py meu_video.mp4")