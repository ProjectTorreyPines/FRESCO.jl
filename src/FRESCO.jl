module FRESCO


using LinearAlgebra
import IMAS
import VacuumFields
using VacuumFields: Green
using FuseUtils: trapz
using StaticArrays: SVector, @SVector
import Interpolations
using PolygonOps: inpolygon, centroid
import HypergeometricFunctions: _₂F₁ as F21
using RecipesBase
using Plots

const μ₀ = π * 4e-7
const twopi = 2π

include("canvas.jl")
include("flux.jl")
include("GSsolve.jl")
include("current.jl")
include("feedback.jl")
include("workflow.jl")

export Canvas
export flux, psinorm
export invert_GS!, invert_GS_zero_bnd!
export initial_current
export solve!

end