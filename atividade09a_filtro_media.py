# =============================================================================
#  ATIVIDADE 09A — Filtro de Média (Low-Pass / Suavização)
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Aplicar o filtro de média para reduzir ruído gaussiano e
#            comparar o efeito de diferentes tamanhos de kernel.
#
#  Fórmula:
#            g(x,y) = (1/(2k+1)²) · ΣΣ f(x+i, y+j)
#
#  onde (2k+1) é o tamanho do kernel. Ex: k=1 → kernel 3x3.
#
#  Kernel 3x3:
#    1/9 × | 1 1 1 |
#           | 1 1 1 |
#           | 1 1 1 |
#
#  Característica: suaviza bordas — quanto maior o kernel, mais borrado.
#  Ideal para: ruído gaussiano (distribuição normal de intensidades).
#  Não ideal para: ruído sal-e-pimenta (use mediana para esse caso).
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH     = "lena.jpg"
KERNELS      = [(3, 3), (5, 5), (7, 7), (15, 15)]   # tamanhos de kernel para comparação
NOISE_STD    = 25        # desvio padrão do ruído gaussiano adicionado artificialmente

# =============================================================================
#  FUNÇÕES AUXILIARES
# =============================================================================
def adicionar_ruido_gaussiano(img_gray: np.ndarray, std: float = 25) -> np.ndarray:
    """Adiciona ruído gaussiano de média 0 e desvio padrão 'std'."""
    ruido = np.random.normal(0, std, img_gray.shape).astype(np.float64)
    resultado = img_gray.astype(np.float64) + ruido
    return np.clip(resultado, 0, 255).astype(np.uint8)

def snr(original: np.ndarray, processada: np.ndarray) -> float:
    """Calcula SNR (Signal-to-Noise Ratio) simplificado."""
    sinal = original.astype(np.float64)
    ruido = (processada.astype(np.float64) - sinal)
    if ruido.std() == 0:
        return float('inf')
    return 20 * np.log10(sinal.mean() / ruido.std())

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
img_noisy = adicionar_ruido_gaussiano(img_gray, NOISE_STD)

print(f"Imagem carregada: {img_gray.shape[1]}x{img_gray.shape[0]}")
print(f"Ruído gaussiano adicionado (std={NOISE_STD})")

# =============================================================================
#  PARTE 1 — Filtro de média em imagem limpa
# =============================================================================
print("\n[1] Filtro de média — comparação de kernels (imagem limpa)")
print(f"  {'Kernel':10}  {'Média':>8}  {'Desvio Padrão':>15}  {'Borrão (estimado)':>20}")
print("  " + "-"*60)
print(f"  {'Original':10}  {img_gray.mean():8.1f}  {img_gray.std():15.1f}")

resultados_limpa = {}
for k in KERNELS:
    suavizada = cv2.blur(img_gray, k)
    resultados_limpa[k] = suavizada
    print(f"  {str(k):10}  {suavizada.mean():8.1f}  {suavizada.std():15.1f}")

# =============================================================================
#  PARTE 2 — Filtro de média em imagem com ruído gaussiano
# =============================================================================
print("\n[2] Filtro de média — remoção de ruído gaussiano")
print(f"  {'Kernel':10}  {'SNR estimado':>15}")
print("  " + "-"*30)
print(f"  {'Ruidosa':10}  (referência)")

resultados_ruidosa = {}
for k in KERNELS:
    restaurada = cv2.blur(img_noisy, k)
    resultados_ruidosa[k] = restaurada
    s = snr(img_gray, restaurada)
    print(f"  {str(k):10}  {s:>15.1f} dB")

# =============================================================================
#  EXIBIÇÃO
# =============================================================================
print("\n[3] Exibindo resultados. Pressione qualquer tecla para avançar.\n")

for nome, img_exib in [("Original", img_gray), ("Com Ruído Gaussiano", img_noisy)]:
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

for k, img_exib in resultados_ruidosa.items():
    nome = f"Média Kernel {k} — remoção ruído"
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

# =============================================================================
#  MOSAICO
# =============================================================================
W = 280
linha1 = np.hstack([legenda(img_gray,  "Original", W), legenda(img_noisy, "Com Ruído (std=25)", W)])
linha2 = np.hstack([legenda(resultados_ruidosa[(3,3)],  "Média 3×3", W),
                    legenda(resultados_ruidosa[(7,7)],  "Média 7×7", W)])
linha3 = np.hstack([legenda(resultados_ruidosa[(5,5)],  "Média 5×5", W),
                    legenda(resultados_ruidosa[(15,15)], "Média 15×15", W)])
mosaico = np.vstack([linha1, linha2, linha3])

cv2.namedWindow("Filtro Média — Atividade 09A", cv2.WINDOW_NORMAL)
cv2.imshow("Filtro Média — Atividade 09A", mosaico)
cv2.imwrite("resultado_09a_mosaico_media.png", mosaico)
print("[✓] Mosaico salvo: resultado_09a_mosaico_media.png")
print("    Pressione qualquer tecla para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("[✓] Atividade 09A concluída!")
