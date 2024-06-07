struct Canvas{T<:Real}
    Rs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int}
    Zs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int}
    Ψ::Matrix{T}
    _a::Vector{T}
    _b::Vector{T}
    _c::Vector{T}
    _MST::Matrix{T}
    _u::Matrix{T}
    _A::Vector{T}
    _B::Vector{T}
    _M::Tridiagonal{T, Vector{T}}
    _S::Matrix{T}
end


function Canvas(dd::IMAS.dd, Nr, Nz=Nr)
    ipl = IMAS.get_build_index(dd.build.layer; type=IMAS._plasma_)
    r, z = dd.build.layer[ipl].outline.r, dd.build.layer[ipl].outline.z
    Rs, Zs = range(minimum(r), maximum(r), Nr), range(minimum(z), maximum(z), Nz)
    return Canvas(Rs, Zs)
end

function Canvas(Rs, Zs)
    Nr, Nz = length(Rs) - 1, length(Zs) - 1
    Ψ = zeros(Nr+1, Nz+1)
    hr = (Rs[end] - Rs[1]) / Nr
    a = @. (1.0 + hr / (2Rs)) ^ -1
    c = @. (1.0 - hr / (2Rs)) ^ -1
    b = a + c
    Ψ = zeros(Nr + 1, Nz + 1)
    u = zero(Ψ)
    A = zero(Rs)
    B = zero(Rs)
    MST = [sqrt(2 / Nz) * sin(π * j * k / Nz) for j in 0:Nz, k in 0:Nz]
    M = Tridiagonal(zeros(Nr), zeros(Nr+1), zeros(Nr))
    S = zero(Ψ)

    return Canvas(Rs, Zs, Ψ, a, b, c, MST, u, A, B, M, S)
end