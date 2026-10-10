"""A3 8P DeepSeek-V4 Flash Eager regression: Muon and AdamW."""
from tests.integration_tests import OverrideDefinitions


def build_test_list() -> list[OverrideDefinitions]:
    common = dict(
        ngpu=8,
        nnodes=1,
        train_script="examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh",
        train_args=("--metrics.enable_tensorboard", "--metrics.log_freq=1"),
        expected_steps=(tuple(range(1, 6)),),
        check_loss=False,
        timeout=7200,
    )
    return [
        OverrideDefinitions(
            test_name="dsv4_flash_a3_8p_example",
            test_descr="DeepSeek-V4 Flash A3 8P Muon Eager, 5 steps",
            override_args=[("--training.steps", "5", "--compile.no-enable")],
            env_vars={
                "ASCEND_SET_ENV_PATH": "/mnt/share/Ascend/20260805101249091/ascend-toolkit/latest/set_env.sh",
                "HF_ASSETS_PATH": "/mnt/share/models/DeepSeek-V4-Flash-bf16",
                "CKPT_INIT_LOAD_PATH": "/mnt/share/dsv4_ckpt_8rank",
            },
            **common,
        ),
        OverrideDefinitions(
            test_name="dsv4_flash_a3_8p_adamw",
            test_descr="DeepSeek-V4 Flash A3 8P AdamW Virtual Optimizer Eager, 5 steps",
            # Preserve the original 4K sequence length. Virtual Optimizer must
            # supply swap-backed AdamW moments; never disable the optimizer
            # override merely to avoid a CPU-only unit-test mismatch.
            override_args=[("--training.steps", "5", "--compile.no-enable",
                            "--optimizer.name", "AdamW")],
            env_vars={
                "ASCEND_SET_ENV_PATH": "/mnt/share/Ascend/20260805101249091/ascend-toolkit/latest/set_env.sh",
                "HF_ASSETS_PATH": "/mnt/share/models/DeepSeek-V4-Flash-bf16",
                "CKPT_INIT_LOAD_PATH": "/mnt/share/dsv4_ckpt_8rank",
                # AdamW needs swap-backed moments to fit the 8P model.
                # Replaces swap_optimizer; preserves the shell's NPU ops.
                "OPTIMIZER_OVERRIDES": "torchtitan_npu.override.common.optimizer.virtual",
            },
            **common,
        ),
    ]
