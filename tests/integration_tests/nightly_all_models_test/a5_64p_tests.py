"""A5 64P DeepSeek-V4 Pro case (eight nodes, currently disabled in scheduler)."""
from tests.integration_tests import OverrideDefinitions

STEPS = 5


def build_test_list() -> list[OverrideDefinitions]:
    return [OverrideDefinitions(
        test_name="dsv4_pro_a5_64p",
        test_descr="DeepSeek-V4 Pro A5 64P, 5 steps",
        ngpu=8, nnodes=8,
        train_script="examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh",
        train_args=("--metrics.enable_tensorboard", "--metrics.log_freq=1"),
        override_args=[("--training.steps", str(STEPS))],
        # A5 is disabled until its actual CANN/HF/checkpoint paths are filled.
        # Keeping them here (not in Lite Actions) prevents false validation.
        env_vars={"ASCEND_SET_ENV_PATH": "", "HF_ASSETS_PATH": "",
                  "CKPT_INIT_LOAD_PATH": ""},
        expected_steps=(tuple(range(1, STEPS + 1)),),
        check_loss=False,
    )]
