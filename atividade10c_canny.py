# =============================================================================
#  ATIVIDADE 10C — Detecção de Bordas com Canny
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Aplicar o algoritmo de Canny e entender seus 4 estágios.
#            Comparar o efeito de diferentes limiares.
#
#  O algoritmo de Canny (1986) é considerado o detector de bordas ótimo.
#  Produz bordas finas, precisas, com baixa taxa de falsos positivos.
#
#  ═══════════════════════════════════════════════════════════════════
#  ESTÁGIOS DO ALGORITMO CANNY:
#
#  1. SUAVIZAÇÃO GAUSSIANA
#     → Remove ruído antes de calcular gradientes
#     → Evita detecção de bordas falsas causadas por ruído
#
#  2. CÁLCULO DO GRADIENTE (Sobel)
#     → Calcula Gx e Gy, magnitude e orientação
#     → Identifica regiões de transição de intensidade
#
#  3. SUPRESSÃO DE NÃO-MÁXIMOS (Non-Maximum Suppression)
#     → Para cada pixel, verifica se é o máximo local na direção do gradiente
#     → Pixels que não são máximos locais são zerados → bordas finas
#
#  4. HISTERESE DE DUPLO LIMIAR (Double Threshold Hysteresis)
#     → Dois limiares: threshold1 (baixo) e threshold2 (alto)
#     → Pixel acima de threshold2 → BORDA FORTE  (sempre incluída)
#     → Pixel entre threshold1 e threshold2 → BORDA FRACA
#       (incluída apenas se conectada a uma borda forte)
#     → Pixel abaixo de threshold1 → DESCARTADO
#  ═══════════════════════════════════════════════════════════════════
#
#  Uso recomendado de limiares:
#    threshold2 ≈ 2× ou 3× threshold1
#    Ex: (50, 150),  (100, 200),  (30, 90)
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH = "lena.jpg"

# Conjuntos de limiares para comparação
THRESHOLD_SETS = [
    (30, 90,  "Baixo — muitas bordas (mais ruído)"),
    (50, 150, "Médio — equilíbrio (padrão)"),
    (100, 200, "Alto — poucas bordas (mais seletivo)"),
    (150, 300, "Muito Alto — bordas mais fortes"),
]

# =============================================================================
#  FUNÇÕES AUXILIARES
# =============================================================================
def legenda(img, texto, w=300):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    res = cv2.resize(img, (w, w))
    cv2.rectangle(res, (0, w - 28), (w, w), (15, 15, 15), -1)
    cv2.putText(res, texto, (5, w - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
    return res

def contar_pixels_borda(img_canny: np.ndarray) -> int:
    """Conta pixels brancos (bordas detectadas)."""
    return int(np.sum(img_canny > 0))

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
#  PARTE 1 — Canny com diferentes limiares
# =============================================================================
print("\n[1] Aplicando Canny com diferentes limiares:")
print(f"  {'threshold1':>12}  {'threshold2':>12}  {'Pixels de borda':>20}  Descrição")
print("  " + "-"*70)

resultados_canny = {}
for t1, t2, desc in THRESHOLD_SETS:
    canny = cv2.Canny(img_gray, threshold1=t1, threshold2=t2)
    n = contar_pixels_borda(canny)
    resultados_canny[(t1, t2)] = canny
    print(f"  {t1:>12}  {t2:>12}  {n:>20}  {desc}")

# =============================================================================
#  PARTE 2 — Influência da suavização gaussiana anterior
# =============================================================================
print("\n[2] Influência da suavização gaussiana anterior ao Canny")

t1_padrao, t2_padrao = 50, 150
configs_blur = [
    (None,       "Sem suavização"),
    ((3, 3), 1.0, "GaussianBlur σ=1.0"),
    ((5, 5), 2.0, "GaussianBlur σ=2.0"),
    ((7, 7), 3.0, "GaussianBlur σ=3.0"),
]

resultados_blur = {}
for cfg in configs_blur:
    if cfg[0] is None:
        img_proc = img_gray
        nome = cfg[1]
    else:
        img_proc = cv2.GaussianBlur(img_gray, cfg[0], cfg[1])
        nome = cfg[2]
    canny = cv2.Canny(img_proc, t1_padrao, t2_padrao)
    n = contar_pixels_borda(canny)
    resultados_blur[nome] = canny
    print(f"  {nome:30}  →  {n} pixels de borda")

# =============================================================================
#  PARTE 3 — Comparação Canny vs Sobel vs Laplaciano
# =============================================================================
print("\n[3] Comparação: Canny vs Sobel vs Laplaciano")

img_blur_comp = cv2.GaussianBlur(img_gray, (3, 3), 1.0)

# Sobel magnitude
sx = cv2.Sobel(img_blur_comp, cv2.CV_16S, 1, 0, ksize=3)
sy = cv2.Sobel(img_blur_comp, cv2.CV_16S, 0, 1, ksize=3)
sobel_mag = np.sqrt(sx.astype(np.float64)**2 + sy.astype(np.float64)**2)
sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)

# Laplaciano
lap = cv2.Laplacian(img_blur_comp, cv2.CV_16S, ksize=3)
lap_abs = cv2.convertScaleAbs(lap)

# Canny
canny_comp = cv2.Canny(img_gray, 50, 150)

# Bordas binarizadas do Sobel e Laplaciano para comparação justa
_, sobel_bin = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)
_, lap_bin   = cv2.threshold(lap_abs,  30, 255, cv2.THRESH_BINARY)

print(f"  Sobel binarizado  (thresh=50)  →  {contar_pixels_borda(sobel_bin)} pixels")
print(f"  Laplaciano binário (thresh=30) →  {contar_pixels_borda(lap_bin)} pixels")
print(f"  Canny (50, 150)               →  {contar_pixels_borda(canny_comp)} pixels")
print(f"  → Canny produz bordas mais finas e contínuas.")

# =============================================================================
#  EXIBIÇÃO
# =============================================================================
print("\n[4] Exibindo resultados. Pressione qualquer tecla para avançar.\n")

exibicoes = [
    ("Original", img_gray),
    ("Canny (30, 90) — muitas bordas",   resultados_canny[(30, 90)]),
    ("Canny (50, 150) — padrão",         resultados_canny[(50, 150)]),
    ("Canny (100, 200) — seletivo",      resultados_canny[(100, 200)]),
    ("Sobel binarizado",                 sobel_bin),
    ("Laplaciano binarizado",            lap_bin),
]
for nome, img_exib in exibicoes:
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

# =============================================================================
#  MOSAICO FINAL
# =============================================================================
W = 300
l1 = np.hstack([legenda(img_gray,                    "Original", W),
                legenda(resultados_canny[(50, 150)],  "Canny (50,150) padrão", W)])
l2 = np.hstack([legenda(resultados_canny[(30, 90)],   "Canny (30,90) mais bordas", W),
                legenda(resultados_canny[(100, 200)],  "Canny (100,200) seletivo", W)])
l3 = np.hstack([legenda(sobel_bin,                    "Sobel binarizado", W),
                legenda(lap_bin,                       "Laplaciano binarizado", W)])
mosaico = np.vstack([l1, l2, l3])

cv2.namedWindow("Canny — Atividade 10C", cv2.WINDOW_NORMAL)
cv2.imshow("Canny — Atividade 10C", mosaico)
cv2.imwrite("resultado_10c_mosaico_canny.png", mosaico)
print("[✓] Mosaico salvo: resultado_10c_mosaico_canny.png")
print("    Pressione qualquer tecla para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("[✓] Atividade 10C concluída!")
print("\n--- EXPERIMENTOS SUGERIDOS ---")
print("  Teste (50, 150) vs (100, 200): observe a diferença de bordas fracas")
print("  Adicione ruído com atividade 09 e veja o impacto no Canny")
print("  Compare com Sobel e Laplaciano: note a espessura das bordas")
