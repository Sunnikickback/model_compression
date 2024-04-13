from onnxruntime.quantization import quantize_dynamic, QuantType
from ..training_utils.training import convert_model_to_onnx
from ..training_utils.utils import get_model, parse_args


def quantize_model(model_path, quantized_model_path, nodes_to_exclude):
    quantize_dynamic(model_path,
                     quantized_model_path,
                     op_types_to_quantize=['MatMul', 'Attention'],
                     weight_type=QuantType.QInt8,
                     per_channel=True,
                     reduce_range=True,
                     nodes_to_exclude=nodes_to_exclude,
                     extra_options={'WeightSymmetric': False, 'MatMulConstBOnly': True})


def main():
    args = parse_args("quantization")
    model, tokenizer, model_config = get_model(args)
    convert_model_to_onnx(model, tokenizer, model_config)
    model_path = f"/models/{model_config.model_type}_{model_config.task_type}.onnx"
    quantized_model_path = f"/models/quantized_{model_config.model_type}_{model_config.task_type}.onnx"
    quantize_model(model_path, quantized_model_path, nodes_to_exclude=args.nodes_to_exclude)


if __name__ == "__main__":
    main()
