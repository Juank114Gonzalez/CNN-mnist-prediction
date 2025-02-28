import pickle
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

# Cargar el modelo entrenado
def load_model(filename="cnn_mnist.pkl"):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

def predict_digit(img):
    img = img.resize((28, 28)).convert("L")  # Convertir a escala de grises y redimensionar
    img_array = np.array(img) / 255.0  # Normalizar
    img_array = img_array.reshape(1, 28, 28, 1)  # Formato adecuado para la CNN
    prediction = model.predict(img_array)
    return np.argmax(prediction)

class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dibujar Número")
        
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()
        self.button_predict = tk.Button(root, text="Predecir", command=self.predict)
        self.button_predict.pack()
        self.button_clear = tk.Button(root, text="Limpiar", command=self.clear)
        self.button_clear.pack()
        
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.label_result = tk.Label(root, text="Resultado: ", font=("Helvetica", 20))
        self.label_result.pack()

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill="black", width=5)
        self.draw.ellipse([x-10, y-10, x+10, y+10], fill="black")
    
    def predict(self):
        img = self.image.copy().convert("L")
        img = img.resize((28, 28))
        result = predict_digit(img)
        self.label_result.config(text=f"Resultado: {result}")
    
    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Resultado: ")

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
