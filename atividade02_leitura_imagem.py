# =============================================================================
#  ATIVIDADE 02 — Leitura e Exibição de Imagens
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Carregar uma imagem do disco, exibir em janela e explorar
#            suas propriedades (dimensões, tipo, canais, pixels).
#
#  Imagem necessária: lena.jpg (ou outra imagem de sua escolha)
#  Coloque a imagem na mesma pasta deste arquivo .py
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH = "lena.jpg"   # altere para o caminho da sua imagem

# =============================================================================
#  PARTE 1 — Leitura da imagem
# =============================================================================
print("\n--- PARTE 1: Leitura da Imagem ---")

# imread retorna uma matriz NumPy (ndarray) com shape (altura, largura, canais)
# Flags disponíveis:
#   cv2.IMREAD_COLOR      → carrega colorido em BGR  (padrão)
#   cv2.IMREAD_GRAYSCALE  → carrega em tons de cinza (1 canal)
#   cv2.IMREAD_UNCHANGED  → preserva canal alpha se houver
img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)

# Verificação — imread retorna None se o arquivo não for encontrado
if img is None:
    print(f"ERRO: não foi possível carregar '{IMG_PATH}'")
    print("Verifique se o arquivo existe na mesma pasta que este script.")
    raise SystemExit(1)

# =============================================================================
#  PARTE 2 — Propriedades da imagem
# =============================================================================
print("\n--- PARTE 2: Propriedades ---")

altura, largura, canais = img.shape
print(f"Dimensões : {largura} x {altura} pixels")
print(f"Canais    : {canais}  (1=cinza, 3=BGR, 4=BGRA)")
print(f"Tipo dado : {img.dtype}  (uint8 = valores 0–255)")
print(f"Pixels    : {img.size}  (altura × largura × canais)")

# Acesso a um pixel específico (linha, coluna) → [B, G, R]
px = img[100, 100]
print(f"\nPixel (linha=100, col=100): B={px[0]}  G={px[1]}  R={px[2]}")
print("⚠  OpenCV usa BGR, não RGB. Cuidado ao misturar com Matplotlib!")

# =============================================================================
#  PARTE 3 — Conversão de espaço de cor
# =============================================================================
print("\n--- PARTE 3: Conversões de cor ---")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print(f"Grayscale shape: {img_gray.shape}  (sem canal, apenas altura×largura)")
print(f"RGB shape      : {img_rgb.shape}")

# =============================================================================
#  PARTE 4 — Exibição
# =============================================================================
print("\n--- PARTE 4: Exibição ---")
print("Pressione qualquer tecla para alternar entre as versões.")

janelas = [
    ("Original (BGR)",    img),
    ("Tons de Cinza",     img_gray),
]

for nome, frame in janelas:
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, frame)
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

# =============================================================================
#  PARTE 5 — Salvamento
# =============================================================================
cv2.imwrite("resultado_02_gray.png", img_gray)
print("\n[✓] Imagem grayscale salva em: resultado_02_gray.png")

cv2.destroyAllWindows()
print("[✓] Atividade 02 concluída!")
