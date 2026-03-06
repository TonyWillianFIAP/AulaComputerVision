# =============================================================================
#  ATIVIDADE 08 — Equalização de Histograma
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Aplicar equalização global e local (CLAHE) de histograma
#            para melhorar o contraste de imagens de baixo contraste.
#
#  Conceito:
#    Histograma h(rk) = nk  →  frequência de cada nível de intensidade rk
#    Normalizado  p(rk) = nk / N  →  probabilidade
#
#    Equalização: sk = (L-1) · Σ p(rj)  para j = 0 até k
#    → redistribui as intensidades para uma distribuição mais uniforme
#
#  Métodos comparados:
#    1. Equalização Global  cv2.equalizeHist()
#       → aplica a mesma transformação em toda a imagem
#       → pode saturar detalhes em regiões de alto contraste
#
#    2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
#       → divide a imagem em blocos (tiles), equaliza localmente
#       → clipLimit impede amplificação excessiva de ruído
#       → bilinear interpolation nas bordas dos blocos evita artefatos
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH = "bird.jpg"   # imagem de baixo contraste ideal para teste
                                 # (tente também: lena.jpg, cameraman.tif)

# Parâmetros do CLAHE
CLIP_LIMIT  = 2.0        # limite de contraste (maior = mais contraste, mais ruído)
TILE_SIZE   = (75, 75)   # tamanho dos blocos para equalização local

# =============================================================================
#  FUNÇÕES AUXILIARES
# =============================================================================
def histograma_imagem(img_gray: np.ndarray, bins: int = 256) -> tuple:
    """Calcula histograma e CDF normalizada de uma imagem em tons de cinza."""
    hist = cv2.calcHist([img_gray], [0], None, [bins], [0, 256])
    hist_norm = hist / (img_gray.shape[0] * img_gray.shape[1])
    cdf = np.cumsum(hist_norm)
    return hist.flatten(), cdf

def adicionar_legenda_img(img_gray: np.ndarray, texto: str,
                           largura: int = 350) -> np.ndarray:
    """Redimensiona, converte para BGR e adiciona barra com legenda."""
    res = cv2.resize(img_gray, (largura, largura))
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(res, (0, largura - 30), (largura, largura), (20, 20, 20), -1)
    cv2.putText(res, texto, (6, largura - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1)
    return res

# =============================================================================
#  CARREGAMENTO
# =============================================================================
img_color = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img_color is None:
    print(f"ERRO: imagem '{IMG_PATH}' não encontrada.")
    print("Dica: qualquer imagem de baixo contraste funciona para este teste.")
    raise SystemExit(1)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
print(f"Imagem carregada: {img_gray.shape[1]}x{img_gray.shape[0]}")

# =============================================================================
#  MÉTODO 1 — Equalização Global
# =============================================================================
print("\n[1] Equalização Global: cv2.equalizeHist()")
img_eq_global = cv2.equalizeHist(img_gray)
print(f"    Original  — média: {img_gray.mean():.1f}   desvio padrão: {img_gray.std():.1f}")
print(f"    Equalizada — média: {img_eq_global.mean():.1f}   desvio padrão: {img_eq_global.std():.1f}")

# =============================================================================
#  MÉTODO 2 — CLAHE (Equalização Local Adaptativa com Limitação de Contraste)
# =============================================================================
print(f"\n[2] CLAHE  clipLimit={CLIP_LIMIT}  tileGridSize={TILE_SIZE}")

clahe = cv2.createCLAHE(clipLimit=CLIP_LIMIT, tileGridSize=TILE_SIZE)
img_clahe = clahe.apply(img_gray)

print(f"    CLAHE — média: {img_clahe.mean():.1f}   desvio padrão: {img_clahe.std():.1f}")

# =============================================================================
#  EXPERIMENTO — Variação do clipLimit
# =============================================================================
print("\n[3] Experimento: variando clipLimit do CLAHE")

clip_values = [1.0, 2.0, 5.0, 10.0, 40.0]
resultados_clip = {}
for clip in clip_values:
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=TILE_SIZE)
    resultados_clip[clip] = c.apply(img_gray)
    print(f"    clipLimit={clip:5.1f}  →  desvio padrão: {resultados_clip[clip].std():.1f}")

# =============================================================================
#  EXIBIÇÃO COMPARATIVA
# =============================================================================
print("\n[4] Exibindo comparação. Pressione qualquer tecla para avançar.\n")

comparacoes = [
    ("Original",            img_gray),
    ("Equalização Global",   img_eq_global),
    (f"CLAHE clip={CLIP_LIMIT}", img_clahe),
]

for nome, img_exib in comparacoes:
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

# =============================================================================
#  MOSAICO TRIPLO
# =============================================================================
DISP = 350
mosaico = np.hstack([
    adicionar_legenda_img(img_gray,       "Original",              DISP),
    adicionar_legenda_img(img_eq_global,  "Equalização Global",    DISP),
    adicionar_legenda_img(img_clahe,      f"CLAHE clip={CLIP_LIMIT}", DISP),
])

cv2.namedWindow("Equalização de Histograma — Atividade 08", cv2.WINDOW_NORMAL)
cv2.imshow("Equalização de Histograma — Atividade 08", mosaico)
cv2.imwrite("resultado_08_mosaico_equalizacao.png", mosaico)
print("[✓] Mosaico salvo: resultado_08_mosaico_equalizacao.png")

# =============================================================================
#  MOSAICO DE VARIAÇÃO DE CLIP LIMIT
# =============================================================================
thumb_clips = [adicionar_legenda_img(img_gray, "Original", 250)]
for clip in clip_values:
    thumb_clips.append(adicionar_legenda_img(resultados_clip[clip], f"clip={clip}", 250))
mosaico_clips = np.hstack(thumb_clips)
cv2.imwrite("resultado_08_variacao_cliplimit.png", mosaico_clips)
print("[✓] Variação clipLimit salva: resultado_08_variacao_cliplimit.png")

print("    Pressione qualquer tecla para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("[✓] Atividade 08 concluída!")
print("\n--- EXPERIMENTOS SUGERIDOS ---")
print("  Altere CLIP_LIMIT: valor baixo (1.0) = pouco contraste; alto (40.0) = muito ruído")
print("  Altere TILE_SIZE: blocos menores = equalização mais local e detalhada")
