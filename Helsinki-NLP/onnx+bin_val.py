# torch==1.12.0+cu113, transformers==4.24.0, onnxruntime==1.9.0, onnxruntime-tools==1.7.0
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import onnxruntime as ort
# import torch
import time
import numpy as np

model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
ort_session = ort.InferenceSession("./onnx/model1/model.onnx")
src_lines = ['Copa de aviones', 'botella de agua caliente', 'pendientes', 'papel de arroz', 'Cubo de rubik', 'trapo', 'máquina de leche de soja', 'Chino', 'encendedor', 'bebe caliente', 'crepe salvador chino', 'olla a presión', 'tianzige', 'rompevientos', 'lavabo', 'lencería', 'almohada', 'moxibustión', 'divertida', 'sobre rojo', 'abrigo de mujer', 'bandeja para hornear electrica', 'Titular de la pluma', 'conjunto de cuatro piezas', 'decoración', 'máquina de leche de soja', 'tinte para el cabello', 'sello', 'secuaces', 'coche', 'lavabo para pies', 'picadora de carne', 'tocado', 'ropa de ninos', 'suéter de mujer', 'bata de baño', 'suéter', 'Alfombras de auto', 'enchufar', 'juguetes sexuales', 'Cubierta de cama antiincrustante de viaje', 'wok', 'chinos', 'sombrero para el sol', 'fideos', 'balde', 'gobernante', 'ropa de mujer', 'caja de jabón', 'Maquina de cafe', 'zapatillas de niños', 'Dinosaurio', 'pantalla', 'Mascara para los ojos', 'Super pegamento', 'pinzas', 'Traje', 'vajilla', 'mancuerna', 'Dominó chino', 'ropa de deporte', 'té', 'maleta', 'cuchara de arroz', 'pantalones', 'zapatillas hombres', 'deflector', 'bolsa de agua caliente', 'lavate la cara', 'seda negra', 'copla', 'medias', 'bandeja', 'cocina de inducción', 'sombrero para el sol', 'ratón', 'libro chino', 'pegatina de doble párpado', 'bragas de mujer', 'golpes', 'té', 'coche', 'Camisa blanca', 'sombrero de copa', 'toalla de limpieza', 'portavasos', 'lápiz', 'llave', 'traje antiguo', 'tejer', 'lindo', 'abrir de nuevo', 'Olla de hierro Zhangqiu', 'Porcelana', 'Cubierta de asiento de coche', 'papel de aluminio', 'bragas ayada', 'caja de palillos', 'lápiz de color', 'medias']

# 根据input_ids生成decoder_input_ids
def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """ Shift input ids one token to the right. """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids

for i in range(100):
    start = time.time()
    for j in range(1):
        # onnx模型的输入为numpy.array
        encoder_inputs = tokenizer(src_lines[i], return_tensors="np")
        decoder_inputs = tokenizer(src_lines[i], return_tensors="np")
        all_inputs_onnx = {
            "input_ids": encoder_inputs["input_ids"],
            "attention_mask": encoder_inputs["attention_mask"],
            "decoder_input_ids": shift_tokens_right(encoder_inputs['input_ids'], model.config.pad_token_id, model.config.decoder_start_token_id),
            "decoder_attention_mask": decoder_inputs["attention_mask"],
        }
        # pytorch模型的输入为torch.Tensor
        inputs = tokenizer(src_lines[i], return_tensors="pt")

        # 两个模型分别进行推理
        outputs_onnx = ort_session.run(["logits"], all_inputs_onnx)[0]
        outputs = model(**inputs, labels=inputs.input_ids)["logits"]

        outputs_onnx = np.argmax(outputs_onnx, axis=-1)
        outputs = np.argmax(outputs.tolist(), axis=-1)

        result_onnx = tokenizer.batch_decode(outputs_onnx, skip_special_tokens=True)
        # print(result_onnx)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(result)
    end = time.time()
    print(i, "time:", (end-start)*100, "ms")
