module FRESCO

using Tensors
using SparseArrays
using LinearAlgebra
using Ferrite
using FerriteGmsh
import IMAS
using NonlinearSolve
using StaticArrays

const μ₀ = π * 4e-7
const twopi = 2π

include("Pinata.jl")

include("mesh.jl")

include("GSsolve.jl")

export create_mesh
export invert_GS

end