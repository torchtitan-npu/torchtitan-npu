import logging
from torch._inductor.lowering import lowerings
from torch_npu._inductor.lowering import _init_set
from torch_npu._inductor.lowering_op_list import GENERATE_LIST, GENERATE_LIST2, FALLBACK_LIST, LOWERING_OVERLOAD_OP

logger = logging.getLogger(__name__)


def fix_npu_inductor():
    lowerings.clear()

    gen_set = set()
    _init_set(GENERATE_LIST, gen_set)
    overload_op_set = set()
    _init_set(LOWERING_OVERLOAD_OP, overload_op_set)

    FALLBACK_LIST.clear()

    logger.info(f"Fix complete. Currernt lowerings: {len(lowerings)}")