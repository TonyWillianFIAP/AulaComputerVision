import cv2
import numpy as np

IMG_PATH = R"C:\Users\labsfiap\Downloads\ImagensTestes\lena.jpg"

def stretching_pixelwise_gray(img, r1, s1, r2, s2):
    altura, largura = img.shape
    resultado = np.zeros((altura, largura), dtype=np.float64)

    a1 = s1 / r1            if r1 != 0   else 0.0
    a2 = (s2 - s1) / (r2 - r1) if r2 != r1  else 0.0
    a3 = (255 - s2) / (255 - r2) if r2 != 255 else 0.0

    for i in range(altura):
        for j in range(largura):
            r = float(img[i, j])
            if r <= r1:
                resultado[i, j] = a1 * r
            elif r <= r2:
                resultado[i, j] = a2 * (r - r1) + s1
            else:
                resultado[i, j] = a3 * (r - r2) + s2

    return np.clip(resultado, 0, 255).astype(np.uint8)

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    # Combinacao A: mais escura
    # r1, s1, r2, s2 = 120, 10, 200, 160
    # Combinacao B: alto contraste
    r1, s1, r2, s2 = 120, 50, 150, 240
    # Combinacao C: mais clara
    # r1, s1, r2, s2 = 135, 10, 135, 248

    img_stretch = stretching_pixelwise_gray(img, r1, s1, r2, s2)

    cv2.namedWindow("window_name", 2)
    cv2.imshow("window_name", img_stretch)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
