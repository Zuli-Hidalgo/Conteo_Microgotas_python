import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
import os
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from scipy import fftpack
from PIL import Image

def show_variable(variable_value):
    showinfo("Total droplets", f"Number of droplets: {variable_value}")

def show_ratio(variable_value):
    showinfo("Ratio", f"Ratio of droplets: {variable_value}")

class ImageSelectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Selector")
        self.master.resizable(False, False)

        self.label = tk.Label(master, text="Choose Image:")
        self.label.pack()

        self.image_path_var = tk.StringVar()
        self.image_path_label = tk.Label(master, textvariable=self.image_path_var)
        self.image_path_label.pack()

        self.select_button = tk.Button(master, text="Select Image", command=self.select_image, font="Helvetica 12", width=35, height=3)
        self.select_button.pack(pady=10)

        self.selected_image_path = None

    def select_image(self):
        image_path = fd.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif")])
        if image_path:
            self.image_path_var.set(image_path)
            self.selected_image_path = image_path

def main():
    root = tk.Tk()
    app = ImageSelectorApp(root)
    root.mainloop()

    path = app.selected_image_path
    if not path:
        print("No image selected.")
        return

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    copia = cv2.resize(img, (1000, 700), interpolation=cv2.INTER_AREA)
    cv2.imshow("Imagen Original", copia)
    cv2.waitKey(0)

    b, g, r = cv2.split(copia)
    enhanced_green = cv2.equalizeHist(g)
    
    _, th = cv2.threshold(enhanced_green, 150, 255, cv2.THRESH_BINARY)#Parametro importante
    #cv2.imshow("a ver", th)

    g2 = cv2.medianBlur(th, 3)
    g2 = cv2.Canny(g2, 0, 10)
    g2 = cv2.dilate(g2, None, iterations=1)
    g2 = cv2.erode(g2, None, iterations=1)

    detected_circles = cv2.HoughCircles(g2, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=5 , minRadius=5, maxRadius=12) #parametro importante
    detected_circles2 = cv2.HoughCircles(g2, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=30, minRadius=3, maxRadius=27)

    count = 0
    tonalidades = []
    if detected_circles is not None and detected_circles2 is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        detected_circles2 = np.uint16(np.around(detected_circles2))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(copia, (a,b), r, (0, 255, 0), 2)
            cv2.circle(copia, (a,b), 1, (0, 0, 255), -1)
            count += 1
            cv2.imshow("All", copia)

        # Obtener la tonalidad de la gota
            tonalidad = np.mean(copia[b - r:b + r, a - r:a + r])
            tonalidades.append(tonalidad)

    count3 = len(tonalidades)
    print("El total de gotas es:", f"{count3}")

    cv2.waitKey(0)
    img2 = cv2.imread(path, cv2.IMREAD_COLOR)
    img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
    copia2 = cv2.resize(img2, (1000, 700), interpolation=cv2.INTER_AREA)

    b3, g3 , r3  = cv2.split(copia2)
    alpha = 1.9  # Contrast control (1.0-3.0)
    beta = 0    # Brightness control (0-100)

    g3 = cv2.convertScaleAbs(g3, alpha=alpha, beta=beta)
    _, th2 = cv2.threshold(g3, 89, 255, cv2.THRESH_BINARY)

    g4 = cv2.medianBlur(th2, 3)
    g4 = cv2.Canny(g4, 0, 10)
    g4 = cv2.dilate(g4, None, iterations=1)
    g4 = cv2.erode(g4, None, iterations=1)
    #cv2.imshow("Alpha & Beta Control", g4)
    #cv2.waitKey(0)

    detected_circles3 = cv2.HoughCircles(g4, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=5, maxRadius=12)
    detected_circles4 = cv2.HoughCircles(g4, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=10, minRadius=3, maxRadius=27)

    count2 = 0
    if detected_circles3 is not None and detected_circles4 is not None:
        detected_circles3 = np.uint16(np.around(detected_circles3))
        detected_circles4 = np.uint16(np.around(detected_circles4))

        for pt in detected_circles3[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(copia2, (a,b), r, (243, 122, 245), 2)
            cv2.circle(copia2, (a,b), 1, (143, 0, 235), 3)
            count2 += 1
            cv2.imshow("Positive", copia2)
    
    
    ratio = count / count2 if count2 != 0 else 0

    show_variable(count)
    show_variable(count2)
    show_ratio(ratio)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Graficar la dispersión de las tonalidades
    plt.figure()
    plt.scatter(range(1, count3 + 1), tonalidades, color='green')
    plt.title('Tonalidades de las Gotas')
    plt.xlabel('Número de Gota')
    plt.ylabel('Tonalidad')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
