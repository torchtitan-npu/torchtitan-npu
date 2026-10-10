"""A3 two-node 16P DeepSeek-V4 Flash AdamW Eager test."""
from tests.integration_tests import OverrideDefinitions

STEPS = 5


def build_test_list() -> list[OverrideDefinitions]:
    return [OverrideDefinitions(
        test_name="dsv4_flash_a3_16p_example",
        test_descr="DeepSeek-V4 Flash two-node 16P AdamW Eager, 5 steps",
        ngpu=8, nnodes=2,
        train_script="examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a3.sh",
        train_args=("--metrics.enable_tensorboard", "--metrics.log_freq=1"),
        override_args=[(
            "--parallelism.expert-parallel-degree", "16",
            "--parallelism.data-parallel-shard-degree", "16",
            "--parallelism.data-parallel-replicate-degree", "1",
            "--training.global-batch-size", "128",
            "--training.steps", str(STEPS),
            "--optimizer.name", "AdamW", "--compile.no-enable",
            "--checkpoint.no-enable", "--debug.moe-force-load-balance",
            "--comm.init-timeout-seconds", "600",
        )],
        # Source Flash recipe owns the full NPU import list.  The shell uses
        # `${OPTIMIZER_OVERRIDES-default}`, so explicit empty removes Muon swap.
        env_vars={
            "ASCEND_SET_ENV_PATH": "/mnt/share/Ascend/20260805101249091/ascend-toolkit/latest/set_env.sh",
            "HF_ASSETS_PATH": "/mnt/share/models/DeepSeek-V4-Flash-bf16",
            "CKPT_INIT_LOAD_PATH": "/mnt/share/dsv4_ckpt_8rank",
            "MASTER_PORT": "6316",
            "HCCL_IF_BASE_PORT": "30160",
            "CONFIG": "deepseek_v4_flash_43layers_16experts",
            "OPTIMIZER_OVERRIDES": "",
            # On some installed torch_npu builds the default Triton backend
            # registers aten.erfc twice at import-time (even in Eager runs).
            # Pin this test's known-good backend, without coupling Lite Actions
            # to the model runtime version.
            "TORCHINDUCTOR_NPU_BACKEND": "ascendc",
        },
        expected_steps=(tuple(range(1, STEPS + 1)),),
        check_loss=False,
    )]
