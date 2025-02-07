import torch
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.overrides import TorchFunctionMode

from .nn import QModuleMixin
from .tensor import QTensor, absmax_scale


__all__ = ["Calibration"]


def _updated_scale(scale, new_scale, momentum):
    if torch.all(scale == 1):
        return new_scale
    return momentum * scale + new_scale * (1.0 - momentum)


class Calibration(TorchFunctionMode):
    """A custom torch dispatch mode to calibrate quantized modules.

    In order to improve the accuracy of the quantized activations, the input and output
    scales of each quantized module are evaluated per-batch using the absmax algorithm and aggregated using a
    momentum.

    The dispatch mode also tracks the calls to each torch function down the model graph, and applies optional
    optimizations:
    - streamline: do not quantize activations that are immediately consumed by an incompatible function (like `add` or `silu`).

    Args:
        momentum (`float`): the momentum to use when updating scales.
        streamline (`bool`): if True, avoid quantizing activations when they are consumed by an incompatible function. Defaults to True.
        debug (`bool`): provide very verbose feedback on the console during calibration.
    """

    def __init__(self, *args, momentum: float = 0.9, streamline=True, debug=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.streamline = streamline
        if streamline:
            self.modules_qactivations = {}
        self.debug = debug

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        qinput = QTensor in types
        output = func(*args, **kwargs)
        if self.streamline and qinput:
            for i, arg in enumerate(args):
                module = getattr(arg, "src_module", None)
                if module is not None:
                    if isinstance(output, QTensor):
                        # Quantized activations are required for that module
                        self.modules_qactivations[module] = True
                    elif isinstance(output, torch.Tensor):
                        # Quantized activations are not required for that module unless another function requires them
                        qactivations_required = self.modules_qactivations.get(module, False)
                        self.modules_qactivations[module] = qactivations_required
        return output

    def __enter__(self):
        super().__enter__()
        self.pre_handle = register_module_forward_pre_hook(self.calibrate_input)
        self.post_handle = register_module_forward_hook(self.calibrate_output)

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.pre_handle.remove()
        self.post_handle.remove()

    def calibrate_input(self, module: torch.nn.Module, input, momentum: float = 0.9):
        if isinstance(module, QModuleMixin) and module.activations is not None:
            input = input[0]
            if isinstance(input, QTensor):
                # Just adopt the maximum scale of the input
                module.input_scale = torch.max(input._scale)
            else:
                # Evaluate the best scale
                input_scale = absmax_scale(input, module.activations)
                module.input_scale = _updated_scale(module.input_scale, input_scale, momentum)
            return input

    def calibrate_output(
        self,
        module: torch.nn.Module,
        input,
        output,
    ):
        if isinstance(module, (QModuleMixin)) and module.activations is not None:
            # Re-evaluate raw module output
            qoutput = module.qforward(input[0])
            if isinstance(qoutput, QTensor):
                qoutput = qoutput.dequantize()
            # Evaluate the optimal scale per-tensor and update output scale
            output_scale = absmax_scale(qoutput, module.activations, axis=None)
            module.output_scale = _updated_scale(module.output_scale, output_scale, self.momentum)
            # Re-evaluate output with the correct output scale
            output = module.forward(input[0])
            if isinstance(output, QTensor):
                # Mark the outputs as generated by this module
                output.src_module = module
            return output
        else:
            if self.streamline:
                for name, child in module.named_children():
                    if isinstance(child, QModuleMixin) and child.activations is not None:
                        qactivations_required = self.modules_qactivations.get(child, False)
                        if not qactivations_required:
                            # Disable activations for this child as its outputs are only consumed by incompatible functions.
                            child.activations = None
            if self.debug:
                for name, child in module.named_children():
                    if isinstance(child, QModuleMixin):
                        classname = child.__class__.__name__
                        trace = f"{name}({classname}) activations are"
                        if child.activations is None:
                            trace += " not quantized."
                        else:
                            trace += f" quantized to {child.activations} with scale {child.output_scale}."
                        print(trace)
