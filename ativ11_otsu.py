import cv2

IMG_PATH = R"cameraman.jpg"

def main():
    img = cv2.imread(IMG_PATH,0)
    if img is None:
        print("Erro ao carregar imagem")
        return
        
    k, img_otsu = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
   
    cv2.namedWindow("original", 2)
    cv2.imshow("original", img)

    cv2.namedWindow("otsu", 2)
    cv2.imshow("otsu", img_otsu)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()