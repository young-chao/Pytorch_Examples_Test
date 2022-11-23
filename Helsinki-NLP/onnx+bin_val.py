from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import onnxruntime as ort
# from optimum.onnxruntime import ORTModelForSeq2SeqLM
import torch
import time
import numpy as np

model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
ort_session = ort.InferenceSession("./onnx/model1/model.onnx")
src_lines = ['Copa de aviones', 'botella de agua caliente', 'pendientes', 'papel de arroz', 'Cubo de rubik', 'trapo', 'máquina de leche de soja', 'Chino', 'encendedor', 'bebe caliente', 'crepe salvador chino', 'olla a presión', 'tianzige', 'rompevientos', 'lavabo', 'lencería', 'almohada', 'moxibustión', 'divertida', 'sobre rojo', 'abrigo de mujer', 'bandeja para hornear electrica', 'Titular de la pluma', 'conjunto de cuatro piezas', 'decoración', 'máquina de leche de soja', 'tinte para el cabello', 'sello', 'secuaces', 'coche', 'lavabo para pies', 'picadora de carne', 'tocado', 'ropa de ninos', 'suéter de mujer', 'bata de baño', 'suéter', 'Alfombras de auto', 'enchufar', 'juguetes sexuales', 'Cubierta de cama antiincrustante de viaje', 'wok', 'chinos', 'sombrero para el sol', 'fideos', 'balde', 'gobernante', 'ropa de mujer', 'caja de jabón', 'Maquina de cafe', 'zapatillas de niños', 'Dinosaurio', 'pantalla', 'Mascara para los ojos', 'Super pegamento', 'pinzas', 'Traje', 'vajilla', 'mancuerna', 'Dominó chino', 'ropa de deporte', 'té', 'maleta', 'cuchara de arroz', 'pantalones', 'zapatillas hombres', 'deflector', 'bolsa de agua caliente', 'lavate la cara', 'seda negra', 'copla', 'medias', 'bandeja', 'cocina de inducción', 'sombrero para el sol', 'ratón', 'libro chino', 'pegatina de doble párpado', 'bragas de mujer', 'golpes', 'té', 'coche', 'Camisa blanca', 'sombrero de copa', 'toalla de limpieza', 'portavasos', 'lápiz', 'llave', 'traje antiguo', 'tejer', 'lindo', 'abrir de nuevo', 'Olla de hierro Zhangqiu', 'Porcelana', 'Cubierta de asiento de coche', 'papel de aluminio', 'bragas ayada', 'caja de palillos', 'lápiz de color', 'medias']

for i in range(100):
    start = time.time()
    for j in range(1):
        encoder_inputs = tokenizer(src_lines[i], return_tensors="np")
        decoder_inputs = tokenizer(src_lines[i], return_tensors="np")
        all_inputs = {
            "input_ids": encoder_inputs["input_ids"],
            "attention_mask": encoder_inputs["attention_mask"],
            "decoder_input_ids": decoder_inputs["input_ids"],
            "decoder_attention_mask": decoder_inputs["attention_mask"],
        }
        inputs = tokenizer(src_lines[i], return_tensors="pt")
        outputs = ort_session.run(["logits"], all_inputs)[0]
        # print(type(outputs))
        outputs0 = model(**inputs, labels=inputs.input_ids)["logits"]
        # print(outputs, outputs0)
        outputs = np.argmax(outputs, axis=-1)
        outputs0 = np.argmax(outputs0.tolist(), axis=-1)
        print(outputs, outputs0)
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(result)
        result0 = tokenizer.batch_decode(outputs0, skip_special_tokens=True)
        print(result0)
    end = time.time()
    print(i, "time:", (end-start)*100, "ms")

def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
