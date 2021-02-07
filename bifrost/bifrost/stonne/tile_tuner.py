"""
-T_R=[x]: Number of flter rows mapped at a time
-T_S=[x]: Number of filter columns mapped at a time
-T_C=[x]: Number of filter and input channels per group mapped at a time
-T_K=[x]: Number of filters and output channels per group mapped at a time
-T_G=[x]: Number of groups mappd at a time
-T_N=[x]: Number of inputs mapped at a time (Only 1 is supported so far)
-T_X_=[x]: Number of input rows mapped at a time
-T_Y_=[x]: Number of input columns mapped a time

Please make sure that these next constraints are followed (i.e., tile dimension must be multiple of its dimension):
If the architecture to be run is flexible (MAERI or SIGMA):
-T_R % R = 0; -T_S % S = 0; -T_C % C = 0 ;-T_K % K = 0; -T_G % G = 0; -T_X_ % ((X - R + strides) / strides) = 0; -T_Y_ % ((Y - S + strides) / strides) = 0;

// Stonne variable taxonomy
// -R: Number of flter rows
// -S: Number of filter columns
// -C: Number of filter and input channels
// -K: Number of filters and output channels
// -G: Number of groups
// -N: Number of inputs (Only 1 is supported so far)
// -X: Number of input rows
// -Y: Number of input columns
// -X_: Number of output columns
// -Y_: Number of output columns
"""

def create_tile_tuning_space(
    R: int,
    S: int,
    C:int,
    K:int,
    G:int,
    X:int,
    Y:int,
    strides:int,
    range_max = 5 
    ):

    # The amount of alternatives to generate
    rn = range(1,range_max+1)

    # Constraints
    X_ =(X - R + strides) // strides
    Y_ = (Y - S + strides) // strides

    # Generate tile configs
    return [
        ("T_R",  [x*R for x in rn]),
        ("T_S",  [x*S for x in rn]),
        ("T_C",  [x*C for x in rn]),
        ("T_K",  [x*K for x in rn]),
        ("T_G",  [x*G for x in rn]),
        ("T_N",  [1]), # Only supported so far
        ("T_X_", [x*X_ for x in rn]),
        ("T_Y_", [x*Y_ for x in rn]),
    ]