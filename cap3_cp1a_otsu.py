import cv2

# Atividade CP1-a — Segmentacao Otsu com e sem suavizacao Gaussiana
# Teste com pelo menos 2 imagens alterando IMG_PATH

IMG_PATH = R"lena.jpg"

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    # --- Otsu SEM suavizacao ---
    k_sem, img_otsu_sem = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Otsu sem gaussiana: k = {k_sem}")

    # --- Otsu COM suavizacao Gaussiana ---
    img_smooth = cv2.GaussianBlur(img, (7, 7), 1.5)
    k_com, img_otsu_com = cv2.threshold(
        img_smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Otsu com gaussiana: k = {k_com}")

    cv2.namedWindow("original", 2)
    cv2.imshow("original", img)

    cv2.namedWindow("otsu sem gaussiana", 2)
    cv2.imshow("otsu sem gaussiana", img_otsu_sem)

    cv2.namedWindow("otsu com gaussiana", 2)
    cv2.imshow("otsu com gaussiana", img_otsu_com)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
