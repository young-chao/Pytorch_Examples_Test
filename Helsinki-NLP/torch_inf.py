# import os
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# print(os.getcwd())

device = "cpu"
cpu_num = 32
torch.set_num_threads(cpu_num)

# f = open("./data/data1")
# src_lines = f.read().splitlines()
src_lines = ['Copa de aviones', 'botella de agua caliente', 'pendientes', 'papel de arroz', 'Cubo de rubik', 'trapo', 'máquina de leche de soja', 'Chino', 'encendedor', 'bebe caliente', 'crepe salvador chino', 'olla a presión', 'tianzige', 'rompevientos', 'lavabo', 'lencería', 'almohada', 'moxibustión', 'divertida', 'sobre rojo', 'abrigo de mujer', 'bandeja para hornear electrica', 'Titular de la pluma', 'conjunto de cuatro piezas', 'decoración', 'máquina de leche de soja', 'tinte para el cabello', 'sello', 'secuaces', 'coche', 'lavabo para pies', 'picadora de carne', 'tocado', 'ropa de ninos', 'suéter de mujer', 'bata de baño', 'suéter', 'Alfombras de auto', 'enchufar', 'juguetes sexuales', 'Cubierta de cama antiincrustante de viaje', 'wok', 'chinos', 'sombrero para el sol', 'fideos', 'balde', 'gobernante', 'ropa de mujer', 'caja de jabón', 'Maquina de cafe', 'zapatillas de niños', 'Dinosaurio', 'pantalla', 'Mascara para los ojos', 'Super pegamento', 'pinzas', 'Traje', 'vajilla', 'mancuerna', 'Dominó chino', 'ropa de deporte', 'té', 'maleta', 'cuchara de arroz', 'pantalones', 'zapatillas hombres', 'deflector', 'bolsa de agua caliente', 'lavate la cara', 'seda negra', 'copla', 'medias', 'bandeja', 'cocina de inducción', 'sombrero para el sol', 'ratón', 'libro chino', 'pegatina de doble párpado', 'bragas de mujer', 'golpes', 'té', 'coche', 'Camisa blanca', 'sombrero de copa', 'toalla de limpieza', 'portavasos', 'lápiz', 'llave', 'traje antiguo', 'tejer', 'lindo', 'abrir de nuevo', 'Olla de hierro Zhangqiu', 'Porcelana', 'Cubierta de asiento de coche', 'papel de aluminio', 'bragas ayada', 'caja de palillos', 'lápiz de color', 'medias']

# model_name = "Helsinki-NLP/opus-mt-es-en" 
model_name = "./model" 
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.src_lang = "es_XX"
# tokenizer.tgt_lang = "en_XX"


model_inputs = ["" for i in range(100)]

print("-------start the inference!-------")
with torch.no_grad():
    for i in range(100):
        # i = 9
        start = time.time()
        generated_tokens = torch.Tensor().to(device)
        dif_time = 0
        for j in range(10):
            model_inputs[i] = tokenizer(src_lines[i], max_length=30, padding=True, return_tensors="pt").to(device)
            model_inputs[i]["num_return_sequences"] = 1
            tem = model.generate(**model_inputs[i], max_new_tokens=30)
            # tem = np.where(tem!=-100, tem, tokenizer.pad_token_id)
            tem_start = time.time()
            generated_tokens = torch.cat( (generated_tokens, tem), dim = 1)
            tem_end =  time.time()
            dif_time += tem_end - tem_start
        result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        end = time.time()
        # print(result[0])
        ti = str((end - start - dif_time)*100) + "\n"
        # print(dif_time)
        print(i, "time:", (end-start-dif_time)*100, "ms")
        with open("./results/time.txt","a") as file:
            file.write(ti)
