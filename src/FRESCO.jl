module FRESCO


using LinearAlgebra
import IMAS

const μ₀ = π * 4e-7
const twopi = 2π

include("canvas.jl")
include("GSsolve.jl")
include("current.jl")

export Canvas
export invert_GS!
export initial_current


end