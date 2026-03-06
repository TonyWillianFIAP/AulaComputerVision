# =============================================================================
#  ATIVIDADE 07 — Realce por Stretching de Contraste (Expansão de Contraste)
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Implementar a transformação de expansão de contraste por meio
#            de uma função linear por partes (piecewise linear).
#
#  Conceito: A transformação é definida por 3 segmentos lineares
#            controlados pelos pontos (r1, s1) e (r2, s2):
#
#   s │            /(r2,s2)──────255
#     │           /
#  s2 │──────────/
#     │         /
#  s1 │────────(r1,s1)
#     │       /
#   0 │──────/
#     └──────────────────── r
#          r1  r2
#
#  Segmento 1  [0, r1]    : s = (s1/r1) * r
#  Segmento 2  [r1, r2]   : s = ((s2-s1)/(r2-r1)) * (r - r1) + s1
#  Segmento 3  [r2, 255]  : s = ((255-s2)/(255-r2)) * (r - r2) + s2
#
#  Combinações de teste (ver tabela no guia):
#    A: r1=120, s1=10,  r2=200, s2=160  → mais escura
#    B: r1=120, s1=50,  r2=150, s2=240  → alto contraste
#    C: r1=135, s1=10,  r2=135, s2=248  → mais clara
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
IMG_PATH = "lena.jpg"

# Escolha a combinação (descomente uma):
r1, s1 = 120, 10   ;  r2, s2 = 200, 160   # Combinação A — mais escura
# r1, s1 = 120, 50  ;  r2, s2 = 150, 240  # Combinação B — alto contraste
# r1, s1 = 135, 10  ;  r2, s2 = 135, 248  # Combinação C — mais clara

# =============================================================================
#  FUNÇÃO DE STRETCHING
# =============================================================================
def stretching_pixelwise_gray(img_gray: np.ndarray,
                               r1: int, s1: int,
                               r2: int, s2: int) -> np.ndarray:
    """
    Aplica transformação de expansão de contraste (piecewise linear)
    pixel a pixel em uma imagem em tons de cinza.

    Parâmetros:
        img_gray : imagem uint8 (0–255) em tons de cinza
        r1, s1   : ponto de controle 1 (entrada, saída)
        r2, s2   : ponto de controle 2 (entrada, saída)

    Retorno:
        Imagem transformada (uint8, 0–255)
    """
    altura, largura = img_gray.shape
    resultado = np.zeros_like(img_gray, dtype=np.float64)

    # Coeficientes dos 3 segmentos lineares
    # Evita divisão por zero com proteção nos denominadores
    a1 = s1 / r1          if r1 != 0          else 0.0
    a2 = (s2 - s1) / (r2 - r1) if r2 != r1   else 0.0
    a3 = (255 - s2) / (255 - r2) if r2 != 255 else 0.0

    print(f"  Coeficiente seg.1 [0, {r1}]       : a1 = {a1:.4f}")
    print(f"  Coeficiente seg.2 [{r1}, {r2}]    : a2 = {a2:.4f}")
    print(f"  Coeficiente seg.3 [{r2}, 255]     : a3 = {a3:.4f}")

    for i in range(altura):
        for j in range(largura):
            r = float(img_gray[i, j])

            if r <= r1:
                resultado[i, j] = a1 * r
            elif r <= r2:
                resultado[i, j] = a2 * (r - r1) + s1
            else:
                resultado[i, j] = a3 * (r - r2) + s2

    return np.clip(resultado, 0, 255).astype(np.uint8)


def stretching_vetorizado(img_gray: np.ndarray,
                           r1: int, s1: int,
                           r2: int, s2: int) -> np.ndarray:
    """
    Versão otimizada (vetorizada com NumPy) — mesma lógica, muito mais rápida.
    Use esta versão para imagens grandes.
    """
    r   = img_gray.astype(np.float64)
    out = np.zeros_like(r)

    a1 = s1 / r1               if r1 != 0   else 0.0
    a2 = (s2 - s1) / (r2 - r1) if r2 != r1  else 0.0
    a3 = (255 - s2) / (255 - r2) if r2 != 255 else 0.0

    out = np.where(r <= r1, a1 * r,
          np.where(r <= r2, a2 * (r - r1) + s1,
                             a3 * (r - r2) + s2))

    return np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================
#  CARREGAMENTO
# =============================================================================
img_color = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
if img_color is None:
    print(f"ERRO: imagem '{IMG_PATH}' não encontrada.")
    raise SystemExit(1)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
print(f"Imagem carregada: {img_gray.shape[1]}x{img_gray.shape[0]}")
print(f"\nParâmetros: r1={r1}  s1={s1}  r2={r2}  s2={s2}")
print()

# =============================================================================
#  APLICAÇÃO
# =============================================================================
print("[1] Aplicando stretching pixel a pixel...")
img_stretch = stretching_pixelwise_gray(img_gray, r1, s1, r2, s2)

print("\n[2] Versão vetorizada (para confirmar equivalência)...")
img_stretch_v = stretching_vetorizado(img_gray, r1, s1, r2, s2)
diff = cv2.absdiff(img_stretch, img_stretch_v).max()
print(f"    Diferença máxima entre versões: {diff}  (deve ser 0)")

# =============================================================================
#  COMPARAÇÃO TODAS AS COMBINAÇÕES
# =============================================================================
print("\n[3] Aplicando todas as combinações de teste...")

combinacoes = {
    "A — Mais escura     (r1=120,s1=10,  r2=200,s2=160)": (120, 10,  200, 160),
    "B — Alto contraste  (r1=120,s1=50,  r2=150,s2=240)": (120, 50,  150, 240),
    "C — Mais clara      (r1=135,s1=10,  r2=135,s2=248)": (135, 10,  135, 248),
}

resultados = {"Original": img_gray}
for desc, (rr1, ss1, rr2, ss2) in combinacoes.items():
    resultados[desc] = stretching_vetorizado(img_gray, rr1, ss1, rr2, ss2)

# =============================================================================
#  EXIBIÇÃO
# =============================================================================
print("\n[4] Exibindo resultados. Pressione qualquer tecla para avançar.\n")
for nome, img_exib in resultados.items():
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    print(f"  {nome[:45]:45s}  média: {img_exib.mean():.1f}")
    cv2.waitKey(0)
    cv2.destroyWindow(nome)

# =============================================================================
#  MOSAICO
# =============================================================================
DISP = 320
def proc(img, texto):
    res = cv2.resize(img, (DISP, DISP))
    res = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(res, (0, DISP - 30), (DISP, DISP), (0, 0, 0), -1)
    cv2.putText(res, texto, (4, DISP - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return res

imgs = [
    proc(img_gray,                           "Original"),
    proc(resultados[list(resultados.keys())[1]], "A: Mais Escura"),
    proc(resultados[list(resultados.keys())[2]], "B: Alto Contraste"),
    proc(resultados[list(resultados.keys())[3]], "C: Mais Clara"),
]
mosaico = np.hstack(imgs)

cv2.namedWindow("Stretching — Atividade 07", cv2.WINDOW_NORMAL)
cv2.imshow("Stretching — Atividade 07", mosaico)
cv2.imwrite("resultado_07_mosaico_stretching.png", mosaico)

print("\n[✓] Mosaico salvo: resultado_07_mosaico_stretching.png")
print("    Pressione qualquer tecla para finalizar.")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("[✓] Atividade 07 concluída!")
