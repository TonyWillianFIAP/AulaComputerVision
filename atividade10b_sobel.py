# =============================================================================
#  ATIVIDADE 10B — Filtro Sobel (High-Pass / Detecção de Bordas Direcional)
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Aplicar o operador Sobel para detecção de bordas em X, Y
#            e calcular a magnitude e orientação do gradiente.
#
#  Conceito: Sobel é um operador de primeira derivada. Combina dois
#            kernels separados para detectar bordas na direção horizontal
#            (Gx) e vertical (Gy).
#
#  Kernels Sobel:
#
#   Gx (bordas verticais):    Gy (bordas horizontais):
#   | -1   0  +1 |            | -1  -2  -1 |
#   | -2   0  +2 |            |  0   0   0 |
#   | -1   0  +1 |            | +1  +2  +1 |
#
#  Magnitude do gradiente:
#            |∇f| = √(Gx² + Gy²)      (magnitude exata)
#            |∇f| ≈ |Gx| + |Gy|        (aproximação rápida)
#
#  Orientação do gradiente:
#            θ = arctan(Gy / Gx)
#
#  Diferença em relação ao Laplaciano:
#    Laplaciano → segunda derivada, isotrópico, detecta qualquer direção
#    Sobel      → primeira derivada, direcional, maior controle por eixo
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH = "figura26a.png"

# =============================================================================
#  FUNÇÕES AUXILIARES
# =============================================================================
def legenda(img, texto, w=300):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    res = cv2.resize(img, (w, w))
    cv2.rectangle(res, (0, w - 28), (w, w), (15, 15, 15), -1)
    cv2.putText(res, texto, (5, w - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1)
    return res

def colorir_orientacao(orientacao_rad: np.ndarray) -> np.ndarray:
    """
    Codifica a orientação do gradiente em cores HSV para visualização.
    Ângulo 0° = vermelho, 90° = verde, 180° = azul, 270° = amarelo.
    """
    h = ((orientacao_rad + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    s = np.full_like(h, 255)
    v = np.full_like(h, 255)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# =============================================================================
#  CARREGAMENTO E PRÉ-PROCESSAMENTO
# =============================================================================
img_color = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img_color is None:
    print(f"ERRO: imagem '{IMG_PATH}' não encontrada.")
    raise SystemExit(1)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
# Suavização prévia para reduzir ruído (recomendado antes do Sobel)
img_blur = cv2.GaussianBlur(img_gray, (3, 3), sigmaX=1.0)
print(f"Imagem carregada: {img_gray.shape[1]}x{img_gray.shape[0]}")

# =============================================================================
#  PARTE 1 — Operador Sobel em X e Y
# =============================================================================
print("\n[1] Calculando gradientes Sobel Gx e Gy...")

# ddepth=cv2.CV_16S mantém sinal; ksize=3 usa kernel 3×3
sobel_x_16 = cv2.Sobel(img_blur, cv2.CV_16S, dx=1, dy=0, ksize=3)
sobel_y_16 = cv2.Sobel(img_blur, cv2.CV_16S, dx=0, dy=1, ksize=3)

# Converte para uint8 para visualização
sobel_x = cv2.convertScaleAbs(sobel_x_16)
sobel_y = cv2.convertScaleAbs(sobel_y_16)

print(f"    Gx — bordas verticais   — max valor: {sobel_x.max()}")
print(f"    Gy — bordas horizontais — max valor: {sobel_y.max()}")

# =============================================================================
#  PARTE 2 — Magnitude do Gradiente
# =============================================================================
print("\n[2] Calculando magnitude do gradiente...")

# Método 1: combinação ponderada com addWeighted
magnitude_pond = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Método 2: √(Gx² + Gy²) — magnitude exata
gx_f = sobel_x_16.astype(np.float64)
gy_f = sobel_y_16.astype(np.float64)
magnitude_exata = np.sqrt(gx_f**2 + gy_f**2)
magnitude_exata = np.clip(magnitude_exata, 0, 255).astype(np.uint8)

# Método 3: |Gx| + |Gy| — aproximação rápida
magnitude_aprox = cv2.convertScaleAbs(
    np.clip(np.abs(sobel_x_16.astype(np.float64)) + np.abs(sobel_y_16.astype(np.float64)), 0, 255)
)

print(f"    Magnitude ponderada (0.5Gx + 0.5Gy) — max: {magnitude_pond.max()}")
print(f"    Magnitude exata √(Gx²+Gy²)          — max: {magnitude_exata.max()}")

# =============================================================================
#  PARTE 3 — Orientação do Gradiente
# =============================================================================
print("\n[3] Calculando orientação do gradiente θ = arctan(Gy/Gx)...")

orientacao = np.arctan2(gy_f, gx_f)   # resultado em radianos [-π, π]
orientacao_vis = colorir_orientacao(orientacao)

print(f"    Orientação — min: {np.degrees(orientacao.min()):.1f}°   max: {np.degrees(orientacao.max()):.1f}°")

# =============================================================================
#  PARTE 4 — Comparação de kernel sizes
# =============================================================================
print("\n[4] Comparação de ksize (1, 3, 5, 7):")
for ks in [1, 3, 5, 7]:
    sx = cv2.Sobel(img_blur, cv2.CV_16S, dx=1, dy=0, ksize=ks)
    sy = cv2.Sobel(img_blur, cv2.CV_16S, dx=0, dy=1, ksize=ks)
    mag = np.sqrt(sx.astype(np.float64)**2 + sy.astype(np.float64)**2)
    print(f"    ksize={ks}  →  bordas detectadas (>50): {np.sum(mag > 50)}")

# =============================================================================
#  EXIBIÇÃO
# =============================================================================
print("\n[5] Exibindo resultados. Pressione qualquer tecla para avançar.\n")

exibicoes = [
    ("Original",                 img_gray),
    ("Sobel Gx (bordas vert.)",  sobel_x),
    ("Sobel Gy (bordas horiz.)", sobel_y),
    ("Magnitude (ponderada)",    magnitude_pond),
    ("Magnitude (exata)",        magnitude_exata),
    ("Orientação (HSV)",         orientacao_vis),
]
for nome, img_exib in exibicoes:
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

# =============================================================================
#  MOSAICO
# =============================================================================
W = 300
l1 = np.hstack([legenda(img_gray,        "Original", W),           legenda(img_blur,         "Pré-blur", W)])
l2 = np.hstack([legenda(sobel_x,         "Sobel Gx (vertical)", W), legenda(sobel_y,          "Sobel Gy (horizontal)", W)])
l3 = np.hstack([legenda(magnitude_exata, "Magnitude √(Gx²+Gy²)", W), legenda(orientacao_vis, "Orientação (HSV)", W)])
mosaico = np.vstack([l1, l2, l3])

cv2.namedWindow("Sobel — Atividade 10B", cv2.WINDOW_NORMAL)
cv2.imshow("Sobel — Atividade 10B", mosaico)
cv2.imwrite("resultado_10b_mosaico_sobel.png", mosaico)
print("[✓] Mosaico salvo: resultado_10b_mosaico_sobel.png")
print("    Pressione qualquer tecla para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("[✓] Atividade 10B concluída!")
