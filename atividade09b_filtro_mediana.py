# =============================================================================
#  ATIVIDADE 09B — Filtro de Mediana (Low-Pass / Não-Linear)
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Aplicar o filtro de mediana para remover ruído sal-e-pimenta
#            (impulse noise) e comparar com o filtro de média.
#
#  Fórmula:
#            g(x,y) = mediana{ f(x+i, y+j) }  para (i,j) na vizinhança
#
#  Característica: substitui o pixel pelo valor mediano da vizinhança.
#    → Não é afetado por valores extremos (pixels totalmente pretos/brancos)
#    → Preserva bordas muito melhor que a média
#    → Ideal para ruído sal-e-pimenta (pixels ruidosos aleatórios)
#
#  Parâmetro: ksize (tamanho do kernel — deve ser ímpar: 3, 5, 7, ...)
#
#  Comparação com Média:
#    Média    → borra bordas, suaviza ruído gaussiano
#    Mediana  → preserva bordas, elimina ruído sal-e-pimenta
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH     = "lena.jpg"   # ideal para ver preservação de bordas
                                  # (tente também lena.jpg, cameraman.tif)
NOISE_FRAC   = 0.05              # 5% dos pixels com ruído sal-e-pimenta
KERNELS      = [3, 5, 7, 11]     # tamanhos de kernel para comparação

# =============================================================================
#  FUNÇÕES AUXILIARES
# =============================================================================
def adicionar_sal_pimenta(img_gray: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """
    Adiciona ruído sal-e-pimenta: frac/2 pixels brancos (255) e frac/2 pretos (0).
    Parâmetros:
        img_gray : imagem de entrada (uint8)
        frac     : fração de pixels corrompidos (0.05 = 5%)
    """
    resultado = img_gray.copy()
    total_pixels = img_gray.size
    n_ruido = int(total_pixels * frac)

    # SAL (branco)
    coords_sal = (np.random.randint(0, img_gray.shape[0], n_ruido // 2),
                  np.random.randint(0, img_gray.shape[1], n_ruido // 2))
    resultado[coords_sal] = 255

    # PIMENTA (preto)
    coords_pimenta = (np.random.randint(0, img_gray.shape[0], n_ruido // 2),
                      np.random.randint(0, img_gray.shape[1], n_ruido // 2))
    resultado[coords_pimenta] = 0

    return resultado

def legenda(img_1ch, texto, w=280):
    res = cv2.resize(img_1ch, (w, w))
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(res, (0, w - 28), (w, w), (15, 15, 15), -1)
    cv2.putText(res, texto, (5, w - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1)
    return res

# =============================================================================
#  CARREGAMENTO E ADIÇÃO DE RUÍDO
# =============================================================================
img_color = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img_color is None:
    print(f"ERRO: imagem '{IMG_PATH}' não encontrada.")
    raise SystemExit(1)

img_gray  = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
img_noisy = adicionar_sal_pimenta(img_gray, NOISE_FRAC)

n_corrompidos = int(img_gray.size * NOISE_FRAC)
print(f"Imagem: {img_gray.shape[1]}x{img_gray.shape[0]}")
print(f"Pixels corrompidos (sal-e-pimenta): {n_corrompidos} ({NOISE_FRAC*100:.0f}%)")

# =============================================================================
#  PARTE 1 — Filtro de Mediana com diferentes kernels
# =============================================================================
print("\n[1] Filtro de Mediana — comparação de kernels")
print(f"  {'Kernel':10}  {'Pixels residuais (0 ou 255)':>30}")
print("  " + "-"*45)

resultados_mediana = {}
for k in KERNELS:
    filtrada = cv2.medianBlur(img_noisy, k)
    resultados_mediana[k] = filtrada
    # Conta pixels que ainda têm valor extremo (ruído residual)
    residual = np.sum((filtrada == 0) | (filtrada == 255))
    print(f"  mediana k={k:2d}     {residual:>30}")

# =============================================================================
#  PARTE 2 — Comparação Mediana vs Média (mesmo kernel)
# =============================================================================
print("\n[2] Comparação: Mediana vs Média (kernel 3×3)")

mediana_3 = cv2.medianBlur(img_noisy, 3)
media_3   = cv2.blur(img_noisy, (3, 3))

# MAE (Mean Absolute Error) em relação à original
mae_mediana = np.mean(np.abs(mediana_3.astype(np.int16) - img_gray.astype(np.int16)))
mae_media   = np.mean(np.abs(media_3.astype(np.int16)   - img_gray.astype(np.int16)))

print(f"  MAE Mediana 3×3 vs Original: {mae_mediana:.2f}  (menor = melhor recuperação)")
print(f"  MAE Média   3×3 vs Original: {mae_media:.2f}")
print(f"  → {'Mediana' if mae_mediana < mae_media else 'Média'} foi mais eficaz neste caso")

# =============================================================================
#  EXIBIÇÃO
# =============================================================================
print("\n[3] Exibindo resultados. Pressione qualquer tecla para avançar.\n")

exibicoes = [
    ("Original",              img_gray),
    ("Com Ruído Sal-e-Pimenta", img_noisy),
    ("Média 3×3 (borrado)",   media_3),
    ("Mediana 3×3 (preserva bordas)", mediana_3),
]
for nome, img_exib in exibicoes:
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

for k, img_exib in resultados_mediana.items():
    nome = f"Mediana k={k}"
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

# =============================================================================
#  MOSAICO
# =============================================================================
W = 280
l1 = np.hstack([legenda(img_gray, "Original", W),
                legenda(img_noisy, "Sal-e-Pimenta 5%", W)])
l2 = np.hstack([legenda(media_3,   "Média 3×3", W),
                legenda(mediana_3,  "Mediana 3×3", W)])
l3 = np.hstack([legenda(resultados_mediana[5], "Mediana 5×5", W),
                legenda(resultados_mediana[7], "Mediana 7×7", W)])

mosaico = np.vstack([l1, l2, l3])
cv2.namedWindow("Filtro Mediana — Atividade 09B", cv2.WINDOW_NORMAL)
cv2.imshow("Filtro Mediana — Atividade 09B", mosaico)
cv2.imwrite("resultado_09b_mosaico_mediana.png", mosaico)
print("[✓] Mosaico salvo: resultado_09b_mosaico_mediana.png")
print("    Pressione qualquer tecla para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("[✓] Atividade 09B concluída!")
