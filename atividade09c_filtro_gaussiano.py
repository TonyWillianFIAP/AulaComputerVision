# =============================================================================
#  ATIVIDADE 09C — Filtro Gaussiano (Low-Pass / Suavização Ponderada)
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Aplicar o filtro gaussiano e comparar com o filtro de média.
#            Demonstrar o efeito do parâmetro sigma na suavização.
#
#  Fórmula do kernel gaussiano 2D:
#
#       G(x, y) = (1 / 2πσ²) · exp( -(x² + y²) / 2σ² )
#
#  onde σ (sigma) controla a "largura" da gaussiana:
#    σ pequeno  → suavização suave, preserva mais detalhes
#    σ grande   → suavização forte, mais borrado
#
#  Diferença em relação ao filtro de média:
#    Média     → peso uniforme para todos os vizinhos
#    Gaussiano → peso maior para vizinhos próximos ao centro
#               (decaimento suave, sem artefatos de borda no kernel)
#    → Gaussiano é a base dos bancos de filtros em CNNs (deep learning)
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH = "lena.jpg"

# Experimentos de sigma (desvio padrão da Gaussiana):
SIGMAS   = [0.5, 1.5, 3.0, 5.0, 10.0]
KSIZE    = (7, 7)   # kernel 7×7 (deve ser ímpar; 0 = OpenCV calcula automaticamente)

# =============================================================================
#  FUNÇÕES AUXILIARES
# =============================================================================
def construir_kernel_gaussiano_manual(ksize: int, sigma: float) -> np.ndarray:
    """
    Constrói manualmente um kernel gaussiano 2D normalizado.
    Útil para entender a construção do filtro.
    """
    k = ksize // 2
    kernel = np.zeros((ksize, ksize), dtype=np.float64)
    for x in range(-k, k + 1):
        for y in range(-k, k + 1):
            kernel[x + k, y + k] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()   # normalização: soma = 1
    return kernel

def legenda(img_1ch, texto, w=280):
    res = cv2.resize(img_1ch, (w, w))
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
#  PARTE 1 — Exibição do kernel gaussiano manual
# =============================================================================
print("\n[1] Kernel Gaussiano 7×7 com sigma=1.5 (construção manual):")
kernel_manual = construir_kernel_gaussiano_manual(7, sigma=1.5)
# Imprime de forma legível
np.set_printoptions(precision=4, suppress=True)
print(kernel_manual)
print(f"    Soma do kernel: {kernel_manual.sum():.6f}  (deve ser ≈ 1.0)")

# =============================================================================
#  PARTE 2 — Filtro gaussiano com OpenCV
# =============================================================================
print("\n[2] Filtro Gaussiano com cv2.GaussianBlur — variando sigma")
print(f"  {'Sigma':>8}  {'Desvio Padrão da Saída':>25}  {'Diferença da Original':>25}")
print("  " + "-"*65)
print(f"  {'Original':>8}  {img_gray.std():25.1f}  -")

resultados_sigma = {}
for sigma in SIGMAS:
    # Quando ksize=(0,0), o tamanho é calculado automaticamente a partir de sigma
    filtrada = cv2.GaussianBlur(img_gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    resultados_sigma[sigma] = filtrada
    diff = cv2.absdiff(img_gray, filtrada).mean()
    print(f"  {sigma:>8}  {filtrada.std():25.1f}  {diff:25.2f}")

# =============================================================================
#  PARTE 3 — Aplicação manual com filter2D e kernel construído à mão
# =============================================================================
print("\n[3] Aplicação manual com cv2.filter2D e kernel gaussiano 7×7 (sigma=1.5)")

kernel_7x7 = construir_kernel_gaussiano_manual(7, sigma=1.5)
# filter2D aplica um kernel arbitrário por convolução
img_manual = cv2.filter2D(img_gray, -1, kernel_7x7.astype(np.float32))

# Comparar com a versão do OpenCV
img_ocv = cv2.GaussianBlur(img_gray, (7, 7), sigmaX=1.5)
diff_manual_ocv = cv2.absdiff(img_manual, img_ocv).max()
print(f"    Diferença máxima manual vs OpenCV: {diff_manual_ocv}  (0 = idênticos)")

# =============================================================================
#  PARTE 4 — Comparação Gaussiano vs Média
# =============================================================================
print("\n[4] Comparação: Gaussiano vs Média (kernel 7×7)")

img_media    = cv2.blur(img_gray, (7, 7))
img_gauss    = cv2.GaussianBlur(img_gray, KSIZE, sigmaX=1.5)

diff_g = cv2.absdiff(img_gray, img_gauss).mean()
diff_m = cv2.absdiff(img_gray, img_media).mean()
print(f"    Diferença média (Gaussiano): {diff_g:.2f}")
print(f"    Diferença média (Média):     {diff_m:.2f}")
print("    O Gaussiano tende a preservar melhor as transições (bordas).")

# =============================================================================
#  EXIBIÇÃO
# =============================================================================
print("\n[5] Exibindo comparação. Pressione qualquer tecla para avançar.\n")

exibicoes = [
    ("Original",              img_gray),
    ("Média 7×7",              img_media),
    (f"Gaussiano 7×7 σ=1.5",  img_gauss),
]
for sigma in [0.5, 3.0, 10.0]:
    exibicoes.append((f"Gaussiano σ={sigma}", resultados_sigma[sigma]))

for nome, img_exib in exibicoes:
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

# =============================================================================
#  MOSAICO
# =============================================================================
W = 280
l1 = np.hstack([legenda(img_gray, "Original", W),
                legenda(img_media, "Média 7×7", W)])
l2 = np.hstack([legenda(resultados_sigma[0.5], "Gaussiano σ=0.5", W),
                legenda(resultados_sigma[1.5], "Gaussiano σ=1.5", W)])
l3 = np.hstack([legenda(resultados_sigma[3.0], "Gaussiano σ=3.0", W),
                legenda(resultados_sigma[10.0], "Gaussiano σ=10.0", W)])

mosaico = np.vstack([l1, l2, l3])
cv2.namedWindow("Filtro Gaussiano — Atividade 09C", cv2.WINDOW_NORMAL)
cv2.imshow("Filtro Gaussiano — Atividade 09C", mosaico)
cv2.imwrite("resultado_09c_mosaico_gaussiano.png", mosaico)
print("[✓] Mosaico salvo: resultado_09c_mosaico_gaussiano.png")
print("    Pressione qualquer tecla para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("[✓] Atividade 09C concluída!")
print("\n--- EXPERIMENTOS SUGERIDOS ---")
print("  σ = 0.5   → suavização mínima, quase imperceptível")
print("  σ = 1.5   → bom balanço (padrão em muitos algoritmos)")
print("  σ = 5.0   → suavização forte, perda de detalhes finos")
