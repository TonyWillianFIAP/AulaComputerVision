import cv2

IMG_PATH = R"lena.jpg"

def main():
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro ao carregar imagem")
        return

    img_canny_1 = cv2.Canny(img, threshold1=50, threshold2=150)
    img_canny_2 = cv2.Canny(img, threshold1=100, threshold2=200)

    cv2.namedWindow("original", 2)
    cv2.imshow("original", img)

    cv2.namedWindow("canny t1=50 t2=150", 2)
    cv2.imshow("canny t1=50 t2=150", img_canny_1)

    cv2.namedWindow("canny t1=100 t2=200", 2)
    cv2.imshow("canny t1=100 t2=200", img_canny_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
