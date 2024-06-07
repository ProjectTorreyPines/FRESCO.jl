module FRESCO


using LinearAlgebra
import IMAS

const μ₀ = π * 4e-7
const twopi = 2π

include("canvas.jl")

include("GSsolve.jl")

export Canvas
export invert_GS!

end