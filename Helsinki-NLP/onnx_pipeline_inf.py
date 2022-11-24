# torch==1.12.0+cu113, transformers==4.24.0, onnxruntime==1.12.0, onnxruntime-tools==1.7.0
import time
import torch
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

cpu_num = 32
torch.set_num_threads(cpu_num)

src_lines = ['Copa de aviones', 'botella de agua caliente', 'pendientes', 'papel de arroz', 'Cubo de rubik', 'trapo', 'máquina de leche de soja', 'Chino', 'encendedor', 'bebe caliente', 'crepe salvador chino', 'olla a presión', 'tianzige', 'rompevientos', 'lavabo', 'lencería', 'almohada', 'moxibustión', 'divertida', 'sobre rojo', 'abrigo de mujer', 'bandeja para hornear electrica', 'Titular de la pluma', 'conjunto de cuatro piezas', 'decoración', 'máquina de leche de soja', 'tinte para el cabello', 'sello', 'secuaces', 'coche', 'lavabo para pies', 'picadora de carne', 'tocado', 'ropa de ninos', 'suéter de mujer', 'bata de baño', 'suéter', 'Alfombras de auto', 'enchufar', 'juguetes sexuales', 'Cubierta de cama antiincrustante de viaje', 'wok', 'chinos', 'sombrero para el sol', 'fideos', 'balde', 'gobernante', 'ropa de mujer', 'caja de jabón', 'Maquina de cafe', 'zapatillas de niños', 'Dinosaurio', 'pantalla', 'Mascara para los ojos', 'Super pegamento', 'pinzas', 'Traje', 'vajilla', 'mancuerna', 'Dominó chino', 'ropa de deporte', 'té', 'maleta', 'cuchara de arroz', 'pantalones', 'zapatillas hombres', 'deflector', 'bolsa de agua caliente', 'lavate la cara', 'seda negra', 'copla', 'medias', 'bandeja', 'cocina de inducción', 'sombrero para el sol', 'ratón', 'libro chino', 'pegatina de doble párpado', 'bragas de mujer', 'golpes', 'té', 'coche', 'Camisa blanca', 'sombrero de copa', 'toalla de limpieza', 'portavasos', 'lápiz', 'llave', 'traje antiguo', 'tejer', 'lindo', 'abrir de nuevo', 'Olla de hierro Zhangqiu', 'Porcelana', 'Cubierta de asiento de coche', 'papel de aluminio', 'bragas ayada', 'caja de palillos', 'lápiz de color', 'medias']
# 路径下有：vocab.json、source.spm、target.spm(、tokenizer_config.json、special_token_map.json)
tokenizer = AutoTokenizer.from_pretrained("./onnx/model0")
# 从onnx文件加载模型，路径下有：config.json、encoder_model.onnx、decoder_with_past_model.onnx、decoder_model.onnx
# from_transformers=True, 则从bin文件加载模型，路径下有config.json、model.bin
model = ORTModelForSeq2SeqLM.from_pretrained("./onnx/model0", from_transformers=False)

for i in range(100):
    start = time.time()
    for j in range(10):
        onnx_translation = pipeline("translation_es_to_en", model=model, tokenizer=tokenizer)
        result = onnx_translation(src_lines[i])
        # if j==0:
        #     print(result[0]["translation_text"])
    end = time.time()
    ti = str((end - start)*100) + "\n"
    print(i, "time:", (end-start)*100, "ms")
