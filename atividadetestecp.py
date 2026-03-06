import cv2
import numpy as np

IMG_PATH     = "lena.jpg"
KERNELS      = [(3, 3), (5, 5), (7, 7), (15, 15)] 

img_color = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)

img_gray  = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

print(f"Imagem carregada: {img_gray.shape[1]}x{img_gray.shape[0]}")


resultados_limpa = {}
for k in KERNELS:
    suavizada = cv2.blur(img_gray, k)
    resultados_limpa[k] = suavizada
    print(f"  {str(k):10}  {suavizada.mean():8.1f}  {suavizada.std():15.1f}")

print("\n[3] Exibindo resultados. Pressione qualquer tecla para avançar.\n")

for nome, img_exib in [("Original", img_gray)]:
    cv2.namedWindow(nome, cv2.WINDOW_NORMAL)
    cv2.imshow(nome, img_exib)
    cv2.waitKey(0)
    cv2.destroyWindow(nome)
