# =============================================================================
#  ATIVIDADE 03 — Leitura e Exibição de Vídeo
#  Disciplina: Visão Computacional e PDI — FIAP
#  Prof. Dr. Paulo Sérgio Rodrigues
# =============================================================================
#
#  Objetivo: Abrir um arquivo de vídeo, processar frame a frame e
#            demonstrar como manipular cada quadro individualmente.
#
#  Vídeo necessário: paisagem01.mp4 (ou outro arquivo .mp4 / .avi)
#  Coloque o vídeo na mesma pasta deste arquivo .py
#
#  Controles durante a reprodução:
#    q  → sair
#    p  → pausar / retomar
#    s  → salvar o frame atual como imagem PNG
# =============================================================================

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
VIDEO_PATH  = "paisagem01.mp4"   # altere para o seu arquivo de vídeo
DELAY_MS    = 25                  # ms entre frames  (~40 fps); aumente para desacelerar

# =============================================================================
#  PARTE 1 — Abertura do vídeo
# =============================================================================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERRO: não foi possível abrir '{VIDEO_PATH}'")
    print("Verifique se o arquivo existe na mesma pasta que este script.")
    raise SystemExit(1)

# Metadados do vídeo
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps          = cap.get(cv2.CAP_PROP_FPS)
largura      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\nVídeo aberto: {VIDEO_PATH}")
print(f"Resolução   : {largura} x {altura} pixels")
print(f"FPS         : {fps:.1f}")
print(f"Total frames: {total_frames}")
print(f"Duração     : {total_frames / fps:.1f} segundos")
print("\nControles: [q] sair  |  [p] pausar  |  [s] salvar frame")

# =============================================================================
#  PARTE 2 — Loop de reprodução frame a frame
# =============================================================================
paused        = False
frame_count   = 0
frames_salvos = 0

cv2.namedWindow("Vídeo — Atividade 03", cv2.WINDOW_NORMAL)

while True:
    if not paused:
        ret, frame = cap.read()

        # ret == False quando o vídeo terminar
        if not ret:
            print("\n[i] Fim do vídeo.")
            break

        frame_count += 1

        # ── Overlay de informações no frame ────────────────────────────────
        info = f"Frame: {frame_count}/{total_frames}  |  {largura}x{altura}  |  [q] sair [p] pausa [s] salvar"
        cv2.rectangle(frame, (0, 0), (len(info) * 8, 22), (0, 0, 0), -1)
        cv2.putText(frame, info, (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("Vídeo — Atividade 03", frame)

    # ── Captura de teclado ──────────────────────────────────────────────────
    key = cv2.waitKey(DELAY_MS) & 0xFF

    if key == ord('q'):
        print("[i] Reprodução interrompida pelo usuário.")
        break

    elif key == ord('p'):
        paused = not paused
        estado = "PAUSADO" if paused else "REPRODUZINDO"
        print(f"[p] {estado} no frame {frame_count}")

    elif key == ord('s') and not paused:
        nome_arquivo = f"frame_{frame_count:05d}.png"
        cv2.imwrite(nome_arquivo, frame)
        frames_salvos += 1
        print(f"[s] Frame salvo: {nome_arquivo}")

# =============================================================================
#  PARTE 3 — Limpeza e resumo
# =============================================================================
cap.release()
cv2.destroyAllWindows()

print(f"\n[✓] Frames processados: {frame_count}")
print(f"[✓] Frames salvos     : {frames_salvos}")
print("[✓] Atividade 03 concluída!")
