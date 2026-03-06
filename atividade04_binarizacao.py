# =============================================================================
#  ATIVIDADE 04 — Acesso a Pixels e Binarização
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Acessar pixels individualmente e implementar a binarização
#            de forma manual (pixel a pixel) e via OpenCV.
#
#  Conceito: Binarização transforma uma imagem em tons de cinza em uma
#            imagem com apenas dois valores: 0 (preto) ou 255 (branco).
#
#  Fórmula:
#            | 255,  se I(x,y) >= L
#  Ĩ(x,y) = |
#            |   0,  se I(x,y)  < L
#
#  onde L é o limiar (threshold).
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH = "lena.jpg"

# Experimente diferentes valores de L:
#   L = 64   → imagem maioritariamente branca (limiar baixo)
#   L = 128  → metade do range (padrão)
#   L = 200  → imagem maioritariamente preta (limiar alto)
L = 128

# =============================================================================
#  PARTE 1 — Carregamento e conversão para cinza
# =============================================================================
img_color = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img_color is None:
    print(f"ERRO: imagem '{IMG_PATH}' não encontrada.")
    raise SystemExit(1)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
altura, largura = img_gray.shape
print(f"Imagem carregada: {largura}x{altura}  |  Limiar L = {L}")

# =============================================================================
#  PARTE 2 — Binarização MANUAL (pixel a pixel)
#            Demonstração do processo explícito com laço for
# =============================================================================
print("\n[2] Binarização manual (pixel a pixel)...")

img_bin_manual = np.zeros((altura, largura), dtype=np.uint8)

for i in range(altura):
    for j in range(largura):
        if img_gray[i, j] >= L:
            img_bin_manual[i, j] = 255
        else:
            img_bin_manual[i, j] = 0

print("    Concluída!")

# =============================================================================
#  PARTE 3 — Binarização com OpenCV (cv2.threshold)
#            Muito mais rápido para imagens grandes
# =============================================================================
print("[3] Binarização com cv2.threshold...")

# retval → limiar usado, thresh → imagem binarizada
retval, img_bin_cv = cv2.threshold(
    img_gray,
    L,             # limiar
    255,           # valor para pixels acima do limiar
    cv2.THRESH_BINARY
)
print(f"    Limiar aplicado: {retval}")

# =============================================================================
#  PARTE 4 — Binarização com Otsu (limiar automático)
#            OpenCV calcula o melhor L automaticamente
# =============================================================================
print("[4] Binarização com método de Otsu (L automático)...")

otsu_val, img_bin_otsu = cv2.threshold(
    img_gray,
    0,             # valor ignorado quando se usa THRESH_OTSU
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print(f"    Limiar calculado por Otsu: {otsu_val:.0f}")

# =============================================================================
#  PARTE 5 — Comparação de resultados
# =============================================================================
print("\n[5] Exibindo resultados. Pressione qualquer tecla para avançar...")

resultados = [
    ("Original Colorida",     img_color),
    ("Tons de Cinza",          img_gray),
    (f"Binária Manual  L={L}", img_bin_manual),
    (f"Binária OpenCV  L={L}", img_bin_cv),
    (f"Binária Otsu    L={otsu_val:.0f}", img_bin_otsu),
]

for nome, img_exib in resultados:
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

# =============================================================================
#  PARTE 6 — Salvamento
# =============================================================================
cv2.imwrite(f"resultado_04_bin_manual_L{L}.png",  img_bin_manual)
cv2.imwrite(f"resultado_04_bin_cv_L{L}.png",      img_bin_cv)
cv2.imwrite(f"resultado_04_bin_otsu_L{int(otsu_val)}.png", img_bin_otsu)

print(f"\n[✓] Imagens salvas com prefixo 'resultado_04_'")
print("[✓] Atividade 04 concluída!")
print("\n--- EXPERIMENTOS SUGERIDOS ---")
print("  Altere o valor de L no início do código e execute novamente:")
print("  L = 64   → predominantemente branca")
print("  L = 128  → equilíbrio (padrão)")
print("  L = 200  → predominantemente preta")
