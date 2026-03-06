# =============================================================================
#  ATIVIDADE 05 — Transformação Gamma (Power-Law)
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Aplicar a transformação gamma para ajustar o brilho e
#            contraste de imagens, especialmente útil em imagens médicas.
#
#  Fórmula:
#            Ĩ(x,y) = c · I(x,y)^γ
#
#  onde:
#    c = constante de escala (geralmente 1.0)
#    γ = parâmetro gamma:
#        γ < 1  → clareia a imagem (mais brilho em tons escuros)
#        γ > 1  → escurece a imagem (acentua tons escuros)
#        γ = 1  → nenhuma alteração (identidade)
#
#  Imagem recomendada: vertebra_mri.jpg  (disponível no GitHub da disciplina)
#  Funciona com qualquer imagem no lugar.
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH = "lena.jpg"   # substitua pelo caminho da sua imagem MRI

# Experimentos de gamma:
#   γ = 0.04  → quase totalmente branca
#   γ = 0.5   → clareia bastante (bom para imagens escuras/subexpostas)
#   γ = 1.0   → sem alteração
#   γ = 2.0   → escurece (bom para imagens superexpostas)
#   γ = 25.0  → quase totalmente preta

GAMMAS = [0.04, 0.25, 0.5, 1.0, 2.0, 5.0, 25.0]

# =============================================================================
#  FUNÇÃO PRINCIPAL DE TRANSFORMAÇÃO GAMMA
# =============================================================================
def aplicar_gamma(imagem_gray: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    """
    Aplica a transformação Power-Law (gamma) a uma imagem em tons de cinza.

    Parâmetros:
        imagem_gray : imagem de entrada (uint8, 0–255, 1 canal)
        gamma       : expoente da transformação
        c           : constante multiplicativa (padrão = 1.0)

    Retorno:
        Imagem transformada (uint8, 0–255)
    """
    # Passo 1: Normaliza para o intervalo [0.0, 1.0]
    img_norm = imagem_gray.astype(np.float64) / 255.0

    # Passo 2: Aplica a fórmula  c · I^γ
    img_gamma = c * np.power(img_norm, gamma)

    # Passo 3: Faz clipping para garantir valores em [0, 1] e reconverte para uint8
    img_gamma = np.clip(img_gamma, 0.0, 1.0)
    img_resultado = (img_gamma * 255).astype(np.uint8)

    return img_resultado


# =============================================================================
#  CARREGAMENTO DA IMAGEM
# =============================================================================
img_color = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img_color is None:
    print(f"ERRO: imagem '{IMG_PATH}' não encontrada.")
    print("Dica: use qualquer imagem .jpg ou .png no lugar.")
    raise SystemExit(1)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
print(f"Imagem carregada: {img_gray.shape[1]}x{img_gray.shape[0]} pixels")

# =============================================================================
#  EXPERIMENTO 1 — Exibição individual para cada gamma
# =============================================================================
print("\n[1] Aplicando diferentes valores de gamma...")
print("    Pressione qualquer tecla para avançar entre os resultados.\n")

for g in GAMMAS:
    resultado = aplicar_gamma(img_gray, gamma=g)

    cv2.namedWindow(f"γ = {g}", cv2.WINDOW_NORMAL)
    cv2.imshow(f"γ = {g}", resultado)
    print(f"    γ = {g:5.2f}  →  média de intensidade: {resultado.mean():.1f}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =============================================================================
#  EXPERIMENTO 2 — Grade comparativa: original + 4 gammas em mosaico
# =============================================================================
print("\n[2] Gerando mosaico comparativo...")

# Redimensiona para exibição lado a lado
h, w = img_gray.shape
DISP_W, DISP_H = 300, 300

def thumb(img):
    return cv2.resize(img, (DISP_W, DISP_H))

# Seleciona 4 gammas representativos para o mosaico
gammas_mosaico = [0.25, 0.5, 2.0, 5.0]
cols = [thumb(img_gray)]
for g in gammas_mosaico:
    cols.append(thumb(aplicar_gamma(img_gray, g)))

# Adiciona legendas
def adicionar_legenda(img_1ch, texto):
    img_bgr = cv2.cvtColor(img_1ch, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_bgr, (0, DISP_H - 26), (DISP_W, DISP_H), (0, 0, 0), -1)
    cv2.putText(img_bgr, texto, (6, DISP_H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img_bgr

legendas = ["Original"] + [f"gamma={g}" for g in gammas_mosaico]
cols_leg = [adicionar_legenda(c, l) for c, l in zip(cols, legendas)]

# Junta horizontalmente
mosaico = np.hstack(cols_leg)

cv2.namedWindow("Comparação Gamma — Atividade 05", cv2.WINDOW_NORMAL)
cv2.imshow("Comparação Gamma — Atividade 05", mosaico)
cv2.imwrite("resultado_05_mosaico_gamma.png", mosaico)

print("[✓] Mosaico salvo: resultado_05_mosaico_gamma.png")
print("    Pressione qualquer tecla para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
#  SALVAMENTO INDIVIDUAL
# =============================================================================
for g in [0.5, 1.0, 2.0]:
    resultado = aplicar_gamma(img_gray, gamma=g)
    nome = f"resultado_05_gamma_{str(g).replace('.', '_')}.png"
    cv2.imwrite(nome, resultado)

print("[✓] Imagens individuais salvas com prefixo 'resultado_05_'")
print("[✓] Atividade 05 concluída!")
