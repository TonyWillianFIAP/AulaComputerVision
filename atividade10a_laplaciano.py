# =============================================================================
#  ATIVIDADE 10A — Filtro Laplaciano (High-Pass / Detecção de Bordas)
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Aplicar o filtro Laplaciano para detecção de bordas e
#            aguçamento (sharpening) de imagens.
#
#  Conceito: O Laplaciano é o operador de segunda derivada isotrópico.
#            Detecta regiões de variação rápida de intensidade (bordas)
#            em todas as direções (independe da orientação).
#
#  Fórmula:
#            ∇²f = ∂²f/∂x² + ∂²f/∂y²
#
#  Kernels Laplacianos típicos:
#
#   4-conectividade:     8-conectividade (mais sensível):
#   |  0  -1   0 |      | -1  -1  -1 |
#   | -1   4  -1 |      | -1   8  -1 |
#   |  0  -1   0 |      | -1  -1  -1 |
#
#  Aguçamento (Sharpening):
#            g(x,y) = f(x,y) + ∇²f(x,y)
#            → soma a imagem com suas bordas para realçar detalhes
#
#  Nota: aplicar suavização gaussiana antes do Laplaciano reduz ruído.
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH = "lena.jpg"
SIGMA_PRE_BLUR = 1.0   # sigma para suavização gaussiana antes do Laplaciano

# =============================================================================
#  FUNÇÕES AUXILIARES
# =============================================================================
def legenda(img, texto, w=300):
    res = cv2.resize(img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (w, w))
    if len(res.shape) == 2:
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(res, (0, w - 28), (w, w), (15, 15, 15), -1)
    cv2.putText(res, texto, (5, w - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1)
    return res

# =============================================================================
#  CARREGAMENTO
# =============================================================================
img_color = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img_color is None:
    print(f"ERRO: imagem '{IMG_PATH}' não encontrada.")
    raise SystemExit(1)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
print(f"Imagem carregada: {img_gray.shape[1]}x{img_gray.shape[0]}")

# =============================================================================
#  PARTE 1 — Suavização prévia (reduz sensibilidade ao ruído)
# =============================================================================
print(f"\n[1] Pré-filtragem gaussiana (σ={SIGMA_PRE_BLUR}) antes do Laplaciano")
img_blur = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=SIGMA_PRE_BLUR)

# =============================================================================
#  PARTE 2 — Filtro Laplaciano com cv2.Laplacian
# =============================================================================
print("[2] Aplicando cv2.Laplacian...")

# CV_16S evita truncamento (valores negativos são mantidos)
lap_16s = cv2.Laplacian(img_blur, cv2.CV_16S, ksize=3)

# convertScaleAbs: |valor| → uint8 (valores negativos viram positivos)
lap_abs = cv2.convertScaleAbs(lap_16s)

print(f"    Laplaciano — min: {lap_16s.min()}  max: {lap_16s.max()}")
print(f"    Após convertScaleAbs — min: {lap_abs.min()}  max: {lap_abs.max()}")

# =============================================================================
#  PARTE 3 — Kernels Laplacianos manuais (4 e 8 conectividade)
# =============================================================================
print("\n[3] Aplicando kernels Laplacianos manualmente")

kernel_4 = np.array([[ 0, -1,  0],
                      [-1,  4, -1],
                      [ 0, -1,  0]], dtype=np.float32)

kernel_8 = np.array([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]], dtype=np.float32)

lap_4 = cv2.convertScaleAbs(cv2.filter2D(img_blur, cv2.CV_16S, kernel_4))
lap_8 = cv2.convertScaleAbs(cv2.filter2D(img_blur, cv2.CV_16S, kernel_8))

print(f"    Laplaciano 4-connect — bordas detectadas: {np.sum(lap_4 > 10)}")
print(f"    Laplaciano 8-connect — bordas detectadas: {np.sum(lap_8 > 10)}")

# =============================================================================
#  PARTE 4 — Aguçamento (Sharpening): g = f + ∇²f
# =============================================================================
print("\n[4] Aguçamento: g = f + Laplaciano")

# Convertemos para int16 para evitar overflow na soma
img_int16 = img_blur.astype(np.int16)

# g = f + ∇²f  (usando Laplaciano com sinal, não o abs)
sharpened = img_int16 + lap_16s
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

# Comparação quantitativa
contraste_original = img_gray.std()
contraste_sharpened = sharpened.std()
print(f"    Desvio padrão original:  {contraste_original:.1f}")
print(f"    Desvio padrão aguçado:   {contraste_sharpened:.1f}")
print(f"    → Aumento de contraste:  +{contraste_sharpened - contraste_original:.1f}")

# =============================================================================
#  PARTE 5 — Exibição
# =============================================================================
print("\n[5] Exibindo resultados. Pressione qualquer tecla para avançar.\n")

exibicoes = [
    ("Original",               img_gray),
    ("Gaussiano (pré-blur)",    img_blur),
    ("Laplaciano (bordas abs)", lap_abs),
    ("Laplaciano 4-conectividade", lap_4),
    ("Laplaciano 8-conectividade", lap_8),
    ("Aguçamento  g = f + ∇²f", sharpened),
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
l1 = np.hstack([legenda(img_gray, "Original", W),    legenda(img_blur, "Pré-blur Gaussiano", W)])
l2 = np.hstack([legenda(lap_4,   "Lap. 4-connect", W), legenda(lap_8,  "Lap. 8-connect", W)])
l3 = np.hstack([legenda(lap_abs, "Laplaciano Abs", W), legenda(sharpened, "Aguçamento", W)])
mosaico = np.vstack([l1, l2, l3])

cv2.namedWindow("Laplaciano — Atividade 10A", cv2.WINDOW_NORMAL)
cv2.imshow("Laplaciano — Atividade 10A", mosaico)
cv2.imwrite("resultado_10a_mosaico_laplaciano.png", mosaico)
print("[✓] Mosaico salvo: resultado_10a_mosaico_laplaciano.png")
print("    Pressione qualquer tecla para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("[✓] Atividade 10A concluída!")
