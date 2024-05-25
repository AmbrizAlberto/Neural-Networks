import numpy as np
import matplotlib.pyplot as plt

def tallohoja(tallos, hojas, frecuencias):
    tallo_hoja = {}
    for tallo, hoja, frecuencia in zip(tallos, hojas, frecuencias):
        if tallo in tallo_hoja:
            tallo_hoja[tallo]['hojas'] += str(hoja)
            tallo_hoja[tallo]['frecuencia'] += frecuencia
        else:
            tallo_hoja[tallo] = {'hojas': str(hoja), 'frecuencia': frecuencia}

    print("Tallo", "Hoja", "Frecuencia")
    print("--------------------------------")
    for tallo, datos in sorted(tallo_hoja.items()):
        print(tallo, datos['hojas'], datos['frecuencia'])

    plt.figure(figsize=(10, 5))  # Ajustar el tamaño de la figura
    plt.barh([str(tallo) for tallo in tallo_hoja.keys()], [datos['frecuencia'] for datos in tallo_hoja.values()]) 
    plt.title('Frecuencia de los tallos')
    plt.xlabel('Frecuencia')
    plt.ylabel('Tallo')
    plt.tight_layout() 
    plt.show()

# Datos de la imagen
tallos = [1, 2, 3, 4]
hojas = [69, 25669, 1112223334445566777788899, 11234577]
frecuencias = [2, 5, 25, 8]

tallohoja(tallos, hojas, frecuencias)