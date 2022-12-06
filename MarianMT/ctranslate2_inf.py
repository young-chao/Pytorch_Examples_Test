# ct2-transformers-converter --model ./model --output_dir ./ct2
# ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-de --output_dir opus-mt-en-de
import time
import ctranslate2
import transformers

# convert后的模型文件目录
translator = ctranslate2.Translator("./ct2")
# 原模型文件目录
tokenizer = transformers.AutoTokenizer.from_pretrained("./model")

src_lines = ['Copa de aviones', 'botella de agua caliente', 'pendientes', 'papel de arroz', 'Cubo de rubik', 'trapo', 'máquina de leche de soja', 'Chino', 'encendedor', 'bebe caliente', 'crepe salvador chino', 'olla a presión', 'tianzige', 'rompevientos', 'lavabo', 'lencería', 'almohada', 'moxibustión', 'divertida', 'sobre rojo', 'abrigo de mujer', 'bandeja para hornear electrica', 'Titular de la pluma', 'conjunto de cuatro piezas', 'decoración', 'máquina de leche de soja', 'tinte para el cabello', 'sello', 'secuaces', 'coche', 'lavabo para pies', 'picadora de carne', 'tocado', 'ropa de ninos', 'suéter de mujer', 'bata de baño', 'suéter', 'Alfombras de auto', 'enchufar', 'juguetes sexuales', 'Cubierta de cama antiincrustante de viaje', 'wok', 'chinos', 'sombrero para el sol', 'fideos', 'balde', 'gobernante', 'ropa de mujer', 'caja de jabón', 'Maquina de cafe', 'zapatillas de niños', 'Dinosaurio', 'pantalla', 'Mascara para los ojos', 'Super pegamento', 'pinzas', 'Traje', 'vajilla', 'mancuerna', 'Dominó chino', 'ropa de deporte', 'té', 'maleta', 'cuchara de arroz', 'pantalones', 'zapatillas hombres', 'deflector', 'bolsa de agua caliente', 'lavate la cara', 'seda negra', 'copla', 'medias', 'bandeja', 'cocina de inducción', 'sombrero para el sol', 'ratón', 'libro chino', 'pegatina de doble párpado', 'bragas de mujer', 'golpes', 'té', 'coche', 'Camisa blanca', 'sombrero de copa', 'toalla de limpieza', 'portavasos', 'lápiz', 'llave', 'traje antiguo', 'tejer', 'lindo', 'abrir de nuevo', 'Olla de hierro Zhangqiu', 'Porcelana', 'Cubierta de asiento de coche', 'papel de aluminio', 'bragas ayada', 'caja de palillos', 'lápiz de color', 'medias']

print("-------start the inference!-------")
for i in range(100):
    start = time.time()
    for j in range(10):
        source = tokenizer.convert_ids_to_tokens(tokenizer.encode(src_lines[i]))
        # num_hypotheses<=beam_size，为最后保留的结果序列数量
        results = translator.translate_batch([source], beam_size=4, num_hypotheses=1)
        target = results[0].hypotheses[0]
        if j==0:
            print(tokenizer.decode(tokenizer.convert_tokens_to_ids(target)))
    end = time.time()
    ti = str(round((end-start)*100, 3)) + "\n"
    # print("*"*10)
    # print(i, "time:", round((end-start)*100, 3), "ms")
    # print("*"*10)
    with open("./results/time.txt","a") as file:
        file.write(ti)
