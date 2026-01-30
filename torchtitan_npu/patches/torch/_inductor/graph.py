import operator
import logging
from typing import Any, Callable

import torch
from torch._decomp import get_decompositions
from torch._inductor import config
from torch._library.utils import get_layout_constraint_tag
from torch._inductor.lowering import (
    constrain_to_fake_tensors,
    FALLBACK_ALLOW_LIST,
    lowerings,
    make_fallback,
    maybe_layout_constraints,
    tag_to_layout_constraint,
)
from torch._inductor.exc import (
    LoweringException,
    MissingOperatorWithDecomp,
    MissingOperatorWithoutDecomp,
)

logger = logging.getLogger(__name__)


def graphlowering_call_function(self, target: Callable, args: Any, 
                                kwargs: dict[str, Any]) -> Any:  # type: ignore[type-arg, override]
    """
    torch._inductor.graph.py

    Special handling for backward ATen ops without layout tags has been removed.
    All operations consistently use the default layout constraint.
    """
    if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
        return torch.fx.Interpreter.call_function(self, target, args, kwargs)

    # hasattr on OpOverloadPacket is slow, check isinstance first
    if not isinstance(target, torch._ops.OpOverloadPacket) and hasattr(
        target, "_inductor_lowering_function"
    ):
        # passthrough lowerings from .pattern_matcher
        return target(*args, **kwargs)

    if target not in lowerings:
        assert isinstance(target, torch._ops.OpOverload), (
            f"{target} is not an OpOverload"
        )
        base_name = target.name().split(".")[0]
        if base_name in FALLBACK_ALLOW_LIST:
            make_fallback(target, warn=False, override_decomp=True)
        elif config.implicit_fallbacks:           
            default_tag: torch._C.Tag = get_layout_constraint_tag(
                target, with_default=True
            )
            decided_constraint = tag_to_layout_constraint(default_tag)

            make_fallback(target, layout_constraint=decided_constraint)

        elif get_decompositions([target]):
            # There isn't a good way to dynamically patch this in
            # since AOT Autograd already ran.  The error message tells
            # the user how to fix it.
            raise MissingOperatorWithDecomp(target, args, kwargs)
        else:
            raise MissingOperatorWithoutDecomp(target, args, kwargs)

    try:
        logger.debug("  via %s", lowerings[target])  # type: ignore[index]

        n = self.current_node
        layout_constraints = maybe_layout_constraints(target)
        if layout_constraints:
            old_args, old_kwargs = args, kwargs
            if layout_constraints is constrain_to_fake_tensors:
                # only constrain_to_fake_tensor if this exists.
                # otherwise, no constraints at all: the implication is
                # that this operator was inserted by a custom pass
                # so we'll give them the freedom.
                if "eager_input_vals" in n.meta:
                    fake_args, fake_kwargs = n.meta["eager_input_vals"]

                    # (fake_args, fake_kwargs) might not align with (args, kwargs).
                    # we need to normalize them based on the schema
                    assert isinstance(target, torch._ops.OpOverload)

                    def normalize(args: Any, kwargs: Any) -> tuple[Any, Any]:
                        result = torch.fx.operator_schemas.normalize_function(
                            target, args, kwargs
                        )
                        assert result is not None
                        return result[0], result[1]

                    fake_args, fake_kwargs = normalize(fake_args, fake_kwargs)
                    args, kwargs = normalize(args, kwargs)
                    old_args, old_kwargs = normalize(old_args, old_kwargs)

                    args, kwargs = constrain_to_fake_tensors(
                        args, kwargs, fake_args, fake_kwargs
                    )
            else:
                args, kwargs = layout_constraints(n, *args, **kwargs)

        out = lowerings[target](*args, **kwargs)  # type: ignore[index]

        if layout_constraints:
            # layout_constraints are allowed to make new copies of the inputs.
            # if they do, and if the target is mutable, then we need to
            # write the new values back into the original inputs.
            self.propagate_mutation(n, old_args, old_kwargs, args, kwargs)  # type: ignore[possibly-undefined]

        return out
    except Exception as e:
        raise LoweringException(e, target, args, kwargs).with_traceback(
            e.__traceback__
        ) from None