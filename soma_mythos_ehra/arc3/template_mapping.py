"""Template-to-Token Mapping — maps each of 104 templates to grammar tokens.

Used by Method 1 to score templates using the 31-token multi-hot predictor.
The mapping is derived statically from the template library structure.
"""
from __future__ import annotations

import torch

from soma_mythos_ehra.arc3.recursive_grammar import TOKEN_VOCAB, NUM_TOKENS


# ---------------------------------------------------------------------------
# Static mapping: each template index -> list of grammar tokens it uses
# ---------------------------------------------------------------------------

# Token indices in TOKEN_VOCAB:
# rotate_90=0, rotate_180=1, rotate_270=2, flip_h=3, flip_v=4, transpose=5,
# scale_2=6, scale_3=7, recolor_map=8, fill_holes=9, flood_fill=10,
# shift_down=11, shift_up=12, shift_left=13, shift_right=14, wrap_h=15, wrap_v=16,
# all=17, by_area_max=18, by_area_min=19, by_density_solid=20, by_density_hollow=21,
# is_largest=22, is_smallest=23, is_solid=24, is_hollow=25, has_area_gt_5=26, has_area_lt_5=27,
# compose=28, apply_to_objects=29, branch=30

# Map DSL primitive names to grammar token names
_PRIM_TO_TOKEN = {
    "rotate": {90: "rotate_90", 180: "rotate_180", 270: "rotate_270"},
    "flip": {"h": "flip_h", "v": "flip_v"},
    "transpose": "transpose",
    "scale": {2: "scale_2", 3: "scale_3"},
    "recolor": "recolor_map",
    "fill_holes": "fill_holes",
    "flood_fill": "flood_fill",
    "shift": "shift_down",
    "wrap": "wrap_h",
    "filter_by_area": {"max": "by_area_max", "min": "by_area_min"},
    "filter_by_density": {"max": "by_density_solid", "min": "by_density_hollow"},
    "filter_by_color": "all",
    "filter_by_size": {"largest": "by_area_max", "smallest": "by_area_min"},
    "filter_by_position": "all",
    "take_n": "all",
    "objects": "all",
    "objects_of_color": "all",
    "recolor_objects": "recolor_map",
    "recolor_by_size": "recolor_map",
    "recolor_by_density": "recolor_map",
    "sort_by_position": "all",
    "sort_by_area": "all",
    "largest_object": "all",
    "smallest_object": "all",
    "object_at": "all",
    "distance_between": "all",
}


def _extract_tokens_from_dsl_node(node) -> list[str]:
    """Extract grammar token names from a DSLNode AST."""
    from soma_mythos_ehra.arc3.dsl_grammar import DSLNode
    tokens = []
    if not isinstance(node, DSLNode):
        return tokens

    prim = node.primitive
    params = node.params

    # Map the primitive to grammar token(s)
    mapping = _PRIM_TO_TOKEN.get(prim)
    if mapping is None:
        pass
    elif isinstance(mapping, str):
        tokens.append(mapping)
    elif isinstance(mapping, dict):
        # Find the param value that selects the variant
        for key in ["mode", "axis", "angle", "factor", "region", "position", "order", "reverse"]:
            if key in params:
                val = params[key]
                if val in mapping:
                    tokens.append(mapping[val])
                    break
        else:
            # Default: take first value
            if mapping:
                tokens.append(next(iter(mapping.values())))

    # Handle composition tokens
    if prim == "compose":
        tokens.append("compose")
    elif prim == "apply_to_objects":
        tokens.append("apply_to_objects")

    # Recurse into children
    for child in node.children:
        tokens.extend(_extract_tokens_from_dsl_node(child))

    return tokens


def build_template_token_matrix(templates: list[tuple[str, object]]) -> torch.Tensor:
    """Build a (num_templates, num_tokens) binary mask matrix.

    Args:
        templates: List of (name, DSLNode) tuples from template library.

    Returns:
        (T, NUM_TOKENS) tensor where matrix[i, j] = 1 if template i uses token j.
    """
    T = len(templates)
    matrix = torch.zeros(T, NUM_TOKENS, dtype=torch.float32)

    for i, (name, prog) in enumerate(templates):
        tokens = _extract_tokens_from_dsl_node(prog)
        seen = set()
        for tok in tokens:
            if tok in TOKEN_VOCAB and tok not in seen:
                matrix[i, TOKEN_VOCAB.index(tok)] = 1.0
                seen.add(tok)

    return matrix


def score_templates_from_grammar_priors(
    grammar_probs: torch.Tensor,
    template_matrix: torch.Tensor,
) -> torch.Tensor:
    """Score templates using grammar token probability vector.

    Args:
        grammar_probs: (NUM_TOKENS,) predicted probabilities from grammar predictor.
        template_matrix: (T, NUM_TOKENS) binary mask from build_template_token_matrix.

    Returns:
        (T,) template scores.
    """
    return template_matrix @ grammar_probs
