# =============================================================================
#  ATIVIDADE 01 — Verificação do Ambiente
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: confirmar que Python, OpenCV e NumPy estão instalados e
#            funcionando corretamente antes de iniciar as demais atividades.
#
#  Como executar:
#    1. Ative o ambiente virtual:  venv\Scripts\activate
#    2. Execute:                   python atividade01_setup.py
# =============================================================================

import sys

# ─── 1. Versão do Python ─────────────────────────────────────────────────────
print("=" * 60)
print("  VERIFICAÇÃO DO AMBIENTE — Visão Computacional / FIAP")
print("=" * 60)

versao_python = sys.version
print(f"\n[1] Python instalado: {versao_python}")

# Alerta se versão não for a recomendada
versao_major, versao_minor, *_ = sys.version_info
if versao_major != 3 or versao_minor != 12:
    print("    ⚠  ATENÇÃO: recomenda-se Python 3.12.x")
    print("       Versão 3.13+ pode ter incompatibilidades com OpenCV.")
else:
    print("    ✓ Versão recomendada (3.12.x)")

# ─── 2. NumPy ─────────────────────────────────────────────────────────────────
try:
    import numpy as np
    print(f"\n[2] NumPy instalado: v{np.__version__}  ✓")
except ImportError:
    print("\n[2] NumPy NÃO encontrado!")
    print("    Execute: pip install numpy")
    sys.exit(1)

# ─── 3. OpenCV ────────────────────────────────────────────────────────────────
try:
    import cv2
    print(f"\n[3] OpenCV instalado: v{cv2.__version__}  ✓")
    print(f"    Build information: {cv2.getBuildInformation()[:120]}...")
except ImportError:
    print("\n[3] OpenCV NÃO encontrado!")
    print("    Execute: pip install opencv-python")
    sys.exit(1)

# ─── 4. Teste básico: criar e exibir imagem sintética ─────────────────────────
print("\n[4] Criando imagem de teste sintética (sem precisar de arquivo)...")

# Cria uma imagem 400x600 com 3 canais (BGR)
img_teste = np.zeros((400, 600, 3), dtype=np.uint8)

# Preenche regiões com cores sólidas para testar os 3 canais
img_teste[0:400, 0:200]   = (255, 0, 0)    # Azul  (B=255, G=0, R=0)
img_teste[0:400, 200:400] = (0, 255, 0)    # Verde (B=0, G=255, R=0)
img_teste[0:400, 400:600] = (0, 0, 255)    # Vermelho (B=0, G=0, R=255)

# Adiciona texto na imagem
cv2.putText(img_teste, "Ambiente OK!",       (165, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
cv2.putText(img_teste, "OpenCV v" + cv2.__version__, (170, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1)
cv2.putText(img_teste, "Pressione qualquer tecla para fechar", (60, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

print("    Dimensões da imagem de teste:", img_teste.shape)
print("    Tipo de dado:", img_teste.dtype)

cv2.namedWindow("FIAP — Verificação do Ambiente", cv2.WINDOW_NORMAL)
cv2.resizeWindow("FIAP — Verificação do Ambiente", 600, 400)
cv2.imshow("FIAP — Verificação do Ambiente", img_teste)

print("\n[✓] Tudo funcionando! Feche a janela ou pressione qualquer tecla.")
print("=" * 60)

cv2.waitKey(0)
cv2.destroyAllWindows()
