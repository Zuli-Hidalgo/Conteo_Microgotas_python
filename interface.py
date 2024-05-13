########################################
'''
class CanvasImage:
    def __init__(self, title = "Image Loader"):
        self.master = tk.Tk()
        self.master.withdraw()
        self.master.title(title)
        self.canvas = tk.Canvas(self.master)
        self.canvas.grid(row = 0, column = 0, sticky = tk.NSEW)
        self.image_button = tk.Button(
            self.master, font = "Helvetica 12",
            text = "Choose Image", command = self.choose_image)
        self.image_button.grid(row = 1, column = 0, sticky = tk.NSEW)
        self.master.update()
        self.master.resizable(False, False)
        self.master.deiconify()
        self.image_path = None  # Variable to store the image path
    def choose_image(self):
        image_name = fd.askopenfilename(title = "Pick your image")
        print(image_name)
        if image_name:
            self.image_path = image_name  # Save the image path to the variable
            self.image = tk.PhotoImage(file = image_name, master = self.master)
            w, h = self.image.width(),self.image.height()
            self.canvas.config(width = w, height = h)
            self.canvas.create_image((0,0), image = self.image, anchor = tk.NW)
        return self.image_path '''
'''
if __name__ == "__main__":
    loader = CanvasImage()
    loader.master.mainloop()'''
'''
loader = CanvasImage()
path = loader.choose_image()
loader.master.mainloop()'''
###########################################################33

from tkinter import messagebox
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import ttk
import os
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from scipy import fftpack
import imutils
from PIL import Image,ImageTk

param2_all = 6
param2_positive = 5
low_threshold_all = 136
low_threshold_positive = 230
min_radius_active = 20      # Valor inicial de Radio mínimo
max_radius_active = 25      # Valor inicial de Radio máximo

# Variables que se ajustan con los sliders
param2_active = param2_all
low_threshold_active = low_threshold_all

def show_variable(variable_value):
    showinfo("Total droplets", f"Number of droplets: {variable_value}")

def show_ratio(variable_value):
    showinfo("Ratio", f"Ratio of droplets: {variable_value}")

def rotate_image():
    global image
    if image is not None:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        display_image()

def display_image():
    if image is not None:
        imageToShow = imutils.resize(image, width=180)
        im = Image.fromarray(imageToShow)
        img = ImageTk.PhotoImage(image=im)
        lblOutputImage.configure(image=img)
        lblOutputImage.image = img
        apply_canny()

def apply_canny():
    global tonalidades
    if image is not None:
        b, g, r = cv2.split(image)
        alpha = 1.9
        beta = 0
        enhanced_green = cv2.convertScaleAbs(g, alpha=alpha, beta=beta)

        _, th = cv2.threshold(enhanced_green, int(low_threshold_active), 255, cv2.THRESH_BINARY)
        g2 = cv2.erode(th, None, iterations=1)

        detected_circles = cv2.HoughCircles(g2, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=int(param2_active), minRadius=int(min_radius_active), maxRadius=int(max_radius_active))
        tonalidades = []
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            result_image = image.copy()

            for i, pt in enumerate(detected_circles[0, :]):
                a, b, r = pt[0], pt[1], pt[2]
                cv2.circle(result_image, (a, b), r, (0, 255, 0), 2)
                tonalidad = np.mean(image[b - r:b + r, a - r:a + r])
                tonalidades.append(tonalidad)

            im = Image.fromarray(result_image)
            img = ImageTk.PhotoImage(image=im)

            lblOutputImage.configure(image=img)
            lblOutputImage.image = img

def plot_droplets():
    if tonalidades:  # Asegura que hay datos para plotear
        plt.figure()
        y = np.arange(len(tonalidades))
        x = tonalidades
        plt.scatter(x, y)
        plt.xlabel('Tonalidad media')
        plt.ylabel('Número de Gota')
        plt.title('Distribución de Tonalidad de las Gotas Detectadas')
        plt.show()
    else:
        messagebox.showinfo("Info", "No hay datos de tonalidades para mostrar.")


def update_params(value):
    global low_threshold_active, param2_active, min_radius_active, max_radius_active
    low_threshold_active = w.get()
    param2_active = x.get()
    min_radius_active = y.get()
    max_radius_active = z.get()
    apply_canny()

def set_all_droplets():
    global param2_active, low_threshold_active, min_radius_active, max_radius_active
    param2_active = param2_all
    low_threshold_active = low_threshold_all
    min_radius_active = 20  # Restablecer a valores predeterminados si es necesario
    max_radius_active = 25
    w.set(low_threshold_all)
    x.set(param2_all)
    y.set(20)
    z.set(25)
    apply_canny()

def set_positive_droplets():
    global param2_active, low_threshold_active, min_radius_active, max_radius_active
    param2_active = param2_positive
    low_threshold_active = low_threshold_positive
    min_radius_active = 20  # Restablecer a valores predeterminados si es necesario
    max_radius_active = 25
    w.set(low_threshold_positive)
    x.set(param2_positive)
    y.set(20)
    z.set(25)
    apply_canny()

def elegir_imagen():
    path_image = fd.askopenfilename(filetypes=[("image", ".jpg"), ("image", ".jpeg"), ("image", ".png")])
    if len(path_image) > 0:
        global image
        image = cv2.imread(path_image)
        image = imutils.resize(image, height=600)
        display_image()  # Mostrar imagen inmediatamente después de cargarla

        imageToShow = imutils.resize(image, width=180)
        im = Image.fromarray(imageToShow)
        img = ImageTk.PhotoImage(image=im)

        lblInputImage.configure(image=img)
        lblInputImage.image = img
        apply_canny()  # Aplicar el filtro inmediatamente de que se carga la imagen

image = None
detected_circles = None

root = tk.Tk()

lblInputImage = tk.Label(root)
lblInputImage.grid(column=0, row=2)

lblOutputImage = tk.Label(root)
lblOutputImage.grid(column=1, row=1, rowspan=12)

btn_rotate = tk.Button(root, text="Rotate Image", command=rotate_image)
btn_rotate.grid(column=1, row=0)  # Botón para rotar la imagen dentro del panel de imagen

btn = tk.Button(root, text="Choose image", width=25, command=elegir_imagen)
btn.grid(column=0, row=0, padx=5, pady=5)

w = tk.Scale(root, from_=0, to=254, resolution=1, orient=tk.HORIZONTAL, label="Umbral", command=update_params)
w.set(low_threshold_active)
w.grid(column=0, row=3, padx=5, pady=5)

x = tk.Scale(root, from_=1, to=30, resolution=1, orient=tk.HORIZONTAL, label="Sensibilidad de detección", command=update_params)
x.set(param2_active)
x.grid(column=0, row=4, padx=5, pady=5)

# Agregar los nuevos sliders para Radio mínimo y Radio máximo
y = tk.Scale(root, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, label="Radio mínimo", command=update_params)
y.set(min_radius_active)
y.grid(column=2, row=1, padx=5, pady=5)  # Posicionados a la derecha de la imagen de salida

z = tk.Scale(root, from_=0, to=200, resolution=1, orient=tk.HORIZONTAL, label="Radio máximo", command=update_params)
z.set(max_radius_active)
z.grid(column=2, row=2, padx=5, pady=5)  # Debajo del slider de Radio mínimo

btn_all = tk.Button(root, text="All Droplets", command=set_all_droplets)
btn_all.grid(column=0, row=5, padx=5, pady=5)

btn_positive = tk.Button(root, text="Positive Droplets", command=set_positive_droplets)
btn_positive.grid(column=0, row=6, padx=5, pady=5)

btn_plot = tk.Button(root, text="Plot Droplets", width=25, command=plot_droplets)
btn_plot.grid(column=0, row=7, padx=5, pady=5)

root.mainloop()

#img = cv2.imread(path, cv2.IMREAD_COLOR)
#img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
#copia = cv2.resize(img, (1000, 700), interpolation=cv2.INTER_AREA)

#######################################################################33
    
'''

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

detected_circles3 = cv2.HoughCircles(g4, cv2.HOUGH_GRADIENT, 1, 15, param1=100, param2=10, minRadius=10, maxRadius=25)

count2 = 0
if detected_circles3 is not None:
    detected_circles3 = np.uint16(np.around(detected_circles3))

    for pt in detected_circles3[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        cv2.circle(copia2, (a,b), r, (243, 122, 245), 2)
        count2 += 1
        cv2.imshow("Positive", copia2)
    
        



        
    
ratio = count / count2 if count2 != 0 else 0

show_variable(count)
show_variable(count2)
show_ratio(ratio)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Graficar la dispersión de las tonalidades
threshold = 68
plt.figure()
plt.scatter(range(1, count3 + 1), tonalidades, color=['green' if y <= threshold else 'red' for y in tonalidades])
plt.title('Tonalidades de las Gotas')
plt.xlabel('Número de Gota')
plt.ylabel('Tonalidad')
plt.grid(True)
plt.show()

# Crear histograma de tonalidades
plt.figure()
# Definir umbrales para categorizar las tonalidades
umbral1 = 20
umbral2 = 70
    
# Separar tonalidades en diferentes categorías basadas en los umbrales
tonalidades_bajas = [tono for tono in tonalidades if tono <= umbral1]
tonalidades_medias = [tono for tono in tonalidades if umbral1 < tono <= umbral2]
tonalidades_altas = [tono for tono in tonalidades if tono > umbral2]
    
# Crear histogramas separados para cada categoría
plt.hist(tonalidades_bajas, bins=30, color='blue', alpha=0.5, label='Tonalidades Bajas')
plt.hist(tonalidades_medias, bins=30, color='green', alpha=0.5, label='Tonalidades Medias')
plt.hist(tonalidades_altas, bins=30, color='red', alpha=0.5, label='Tonalidades Altas')

plt.title('Histograma de Tonalidades de las Gotas')
plt.xlabel('Tonalidad')
plt.ylabel('Número de Gotas')
plt.legend()
plt.grid(True)
plt.show()
'''
