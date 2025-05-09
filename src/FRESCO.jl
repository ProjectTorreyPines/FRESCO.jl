module FRESCO


using LinearAlgebra
import IMAS
import VacuumFields
import IMASutils
using IMASutils: trapz
using StaticArrays: SVector, @SVector, @SMatrix
import Interpolations
import DataInterpolations: DataInterpolations, ExtrapolationType
using PolygonOps: inpolygon, centroid
import HypergeometricFunctions: _₂F₁ as F21
using RecipesBase
using Plots
using LoopVectorization: @turbo, @tturbo

const μ₀ = π * 4e-7
const twopi = 2π

include("canvas.jl")
include("current.jl")
include("flux.jl")
include("flux_surfaces.jl")
include("GSsolve.jl")
include("feedback.jl")
include("workflow.jl")

export Canvas
export psinorm
export invert_GS!, invert_GS_zero_bnd!
export initial_current
export solve!

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__, all=false, imported=false) if name != Symbol(@__MODULE__)]

end
