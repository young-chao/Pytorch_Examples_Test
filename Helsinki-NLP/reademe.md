# 将torch模型转为onnx模型
python -m transformers.onnx --model=./model  --feature=seq2seq-lm --atol=1e-4 onnx/model1
# onnx模型简化
python -m onnxsim ./model1/model.onnx ./model2/model.onnx
# onnx模型转IR模型格式
mo --input_model model.onnx --output_dir model_xml
# IR模型推理测试
benchmark_app -shape [1,6] -hint none -t 10 -nstreams 1  -nthreads 32 -m ./model_xml/model.xml
