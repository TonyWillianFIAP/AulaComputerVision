# =============================================================================
#  ATIVIDADE 06 — Transformação Negativo
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Implementar a transformação de negativo de imagem, útil para
#            realçar detalhes em regiões escuras (ex: mamografias).
#
#  Fórmula (imagem normalizada [0,1]):
#            Ĩ(x,y) = 1 - I(x,y)
#
#  Equivalente em uint8 (0–255):
#            Ĩ(x,y) = 255 - I(x,y)
#
#  Aplicação médica: em mamografias, lesões aparecem como regiões escuras.
#  O negativo as torna claras, facilitando a detecção visual.
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH = "lena.jpg"     # funciona com qualquer imagem (tente mamografia.jpg)

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
#  MÉTODO 1 — Negativo via operação matemática (explícito)
# =============================================================================
print("\n[1] Negativo matemático: Ĩ = 255 - I")

# Garante uint8 para evitar overflow
neg_matematico = (255 - img_gray.astype(np.int16)).astype(np.uint8)

# Versão equivalente com normalização para [0,1]:
# img_norm = img_gray.astype(np.float64) / 255.0
# neg_norm = 1.0 - img_norm
# neg_matematico = (neg_norm * 255).astype(np.uint8)

# =============================================================================
#  MÉTODO 2 — Negativo com cv2.bitwise_not (otimizado, recomendado)
# =============================================================================
print("[2] Negativo com cv2.bitwise_not")

# bitwise_not inverte todos os bits:  ~valor = 255 - valor para uint8
neg_bitwise = cv2.bitwise_not(img_gray)

# =============================================================================
#  VERIFICAÇÃO — os dois métodos devem produzir resultado idêntico
# =============================================================================
diferenca = cv2.absdiff(neg_matematico, neg_bitwise)
print(f"\n[✓] Diferença máxima entre os métodos: {diferenca.max()}")
print("    (0 = idênticos, como esperado)")

# =============================================================================
#  NEGATIVO EM IMAGEM COLORIDA (aplica em todos os canais)
# =============================================================================
print("[3] Negativo em imagem colorida (BGR)")
neg_colorido = cv2.bitwise_not(img_color)

# =============================================================================
#  EXIBIÇÃO COMPARATIVA
# =============================================================================
print("\n[4] Exibindo comparação. Pressione qualquer tecla para avançar.\n")

pares = [
    ("Original — Cinza",     img_gray),
    ("Negativo — Cinza",     neg_bitwise),
    ("Original — Colorida",  img_color),
    ("Negativo — Colorida",  neg_colorido),
]

for nome, img_exib in pares:
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    media = img_exib.mean()
    print(f"  {nome:28s}  média intensidade: {media:.1f}")
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

# =============================================================================
#  MOSAICO 2x2
# =============================================================================
DISP = 350
def resize_gray_to_bgr(img, size):
    res = cv2.resize(img, (size, size))
    if len(res.shape) == 2:
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    return res

def legenda(img_bgr, texto):
    out = img_bgr.copy()
    cv2.rectangle(out, (0, DISP - 26), (DISP, DISP), (0, 0, 0), -1)
    cv2.putText(out, texto, (6, DISP - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out

tl = legenda(resize_gray_to_bgr(img_gray,   DISP), "Original Cinza")
tr = legenda(resize_gray_to_bgr(neg_bitwise, DISP), "Negativo Cinza")
bl = legenda(resize_gray_to_bgr(img_color,  DISP), "Original Colorida")
br = legenda(resize_gray_to_bgr(neg_colorido, DISP),"Negativo Colorido")

mosaico = np.vstack([np.hstack([tl, tr]), np.hstack([bl, br])])

cv2.namedWindow("Negativo — Atividade 06", cv2.WINDOW_NORMAL)
cv2.imshow("Negativo — Atividade 06", mosaico)
cv2.imwrite("resultado_06_mosaico_negativo.png", mosaico)
print("\n[✓] Mosaico salvo: resultado_06_mosaico_negativo.png")
print("    Pressione qualquer tecla para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("[✓] Atividade 06 concluída!")
