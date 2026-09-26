"""Explicit device selection and execution metadata for experiment entry points."""

from __future__ import annotations

import os
import platform
import sys


def configure_device(device="cpu"):
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be cpu or gpu")
    if "tensorflow" in sys.modules:
        raise RuntimeError("Select the device before importing TensorFlow")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if device == "cpu" else "0"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import tensorflow as tf

    tf.config.experimental.enable_tensor_float_32_execution(False)
    gpus = tf.config.list_physical_devices("GPU")
    if device == "gpu" and not gpus:
        raise RuntimeError("GPU requested, but TensorFlow found no compatible GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return {
        "device": device.upper(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
        "tf32_enabled": False,
        "gpu_devices": [tf.config.experimental.get_device_details(gpu).get("device_name", gpu.name) for gpu in gpus],
        "intra_op_threads": tf.config.threading.get_intra_op_parallelism_threads(),
        "inter_op_threads": tf.config.threading.get_inter_op_parallelism_threads(),
        "thread_count_note": "Zero means the TensorFlow runtime default.",
    }
