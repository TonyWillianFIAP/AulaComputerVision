import cv2

IMG_PATH = R"C:\Users\labsfiap\Downloads\ImagensTestes\lena.jpg"

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    # Experimente: (3,3), (5,5), (7,7)
    img_media_3 = cv2.blur(img, (3, 3))
    img_media_5 = cv2.blur(img, (5, 5))
    img_media_7 = cv2.blur(img, (7, 7))

    cv2.namedWindow("original", 2)
    cv2.imshow("original", img)

    cv2.namedWindow("media 3x3", 2)
    cv2.imshow("media 3x3", img_media_3)

    cv2.namedWindow("media 5x5", 2)
    cv2.imshow("media 5x5", img_media_5)

    cv2.namedWindow("media 7x7", 2)
    cv2.imshow("media 7x7", img_media_7)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
