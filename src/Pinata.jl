mutable struct Pinata{T<:Real, M<:Ferrite.Grid}
    mesh :: M
    Ψ :: Vector{T}
    dh :: Ferrite.DofHandler{2, M}
    cellvalues::Ferrite.CellScalarValues{2, T, RefTetrahedron}
    facevalues::Ferrite.FaceScalarValues{2, T, RefTetrahedron}
    K::Symmetric{T, SparseMatrixCSC{T, Int}}
    f::Vector{T}
    Ke::Matrix{T}
    fe::Vector{T}
    global_dofs::Vector{Int}
    ∂Ω::Set{FaceIndex}
end

function flux(pin::Pinata, q_point::Int, global_dofs::Vector{Int})
    return function_value(pin.cellvalues, q_point, pin.Ψ, global_dofs)
end

function find_minimum_Ψbnd(pin::Pinata; global_dofs::Vector{Int} = zeros(Int, ndofs_per_cell(pin.dh)), cache::CellCache = CellCache(pin.dh))
    minval = Inf
    for (cellid, faceid) in pin.∂Ω
        reinit!(cache, cellid)
        coords = getcoordinates(cache)
        celldofs!(global_dofs, cache)
        reinit!(pin.facevalues, coords, faceid)
        for q_point in 1:getnquadpoints(pin.facevalues)
            val  = function_value(pin.facevalues, q_point, pin.Ψ, global_dofs)
            (val < minval) && (minval = val)
        end
    end
    return minval
end
