module FRESCO


using LinearAlgebra
import IMAS
using VacuumFields: Green
using FuseUtils: trapz

const μ₀ = π * 4e-7
const twopi = 2π

include("canvas.jl")
include("flux.jl")
include("GSsolve.jl")
include("current.jl")

export Canvas
export flux, psinorm
export invert_GS!, invert_GS_zero_bnd!
export initial_current


end