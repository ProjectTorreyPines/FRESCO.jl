const CoilVectorType = AbstractVector{<:Union{VacuumFields.AbstractCoil, IMAS.pf_active__coil, IMAS.pf_active__coil___element}}

mutable struct Canvas{T<:Real, VC<:CoilVectorType, I<:IMAS.Interpolations.AbstractInterpolation}
    Rs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int}
    Zs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int}
    Ψ::Matrix{T}
    Ip::T
    coils::VC
    Raxis::T
    Zaxis::T
    Ψaxis::T
    Ψbnd::T
    _Ψpl::Matrix{T}
    _Ψvac::Matrix{T}
    _U::Matrix{T}
    _Jt::Matrix{T}
    _Ψitp::I
    _bnd::Vector{Tuple{T,T}}
    _rextrema::Tuple{T,T}
    _zextrema::Tuple{T,T}
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

    # define grid
    layer = dd.build.layer
    k = IMAS.get_build_index(layer; type=IMAS._plasma_)
    r, z = layer[k].outline.r, layer[k].outline.z
    Rs, Zs = range(minimum(r), maximum(r), Nr), range(minimum(z), maximum(z), Nz)

    # define current
    eqt = dd.equilibrium.time_slice[]
    Ip = eqt.global_quantities.ip

    # define coils
    coils = VacuumFields.IMAS_pf_active__coils(dd)

    return Canvas(Rs, Zs, Ip, coils)
end

function Canvas(Rs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int},
                Zs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int},
                Ip::T,
                coils::CoilVectorType) where {T <: Real}
    Nr, Nz = length(Rs) - 1, length(Zs) - 1
    hr = (Rs[end] - Rs[1]) / Nr
    a = @. (1.0 + hr / (2Rs)) ^ -1
    c = @. (1.0 - hr / (2Rs)) ^ -1
    b = a + c
    Ψ = zeros(T, Nr + 1, Nz + 1)
    Ψpl = zero(Ψ)
    Ψvac = zero(Ψ)
    U = zero(Ψ)
    Jt = zero(Ψ)
    Ψitp = IMAS.ψ_interpolant(Rs, Zs, Ψ).PSI_interpolant
    u = zero(Ψ)
    A = zero(Rs)
    B = zero(Rs)
    MST = [sqrt(2 / Nz) * sin(π * j * k / Nz) for j in 0:Nz, k in 0:Nz]
    M = Tridiagonal(zeros(T, Nr), zeros(T, Nr+1), zeros(T, Nr))
    S = zero(Ψ)
    zt = zero(T)
    return Canvas(Rs, Zs, Ψ, Ip, coils, zt, zt, zt, zt, Ψpl, Ψvac, U, Jt, Ψitp, Tuple{T, T}[], (0.0, 0.0), (0.0, 0.0), a, b, c, MST, u, A, B, M, S)
end

function update_interpolation!(canvas::Canvas)
    canvas._Ψitp = IMAS.ψ_interpolant(canvas.Rs, canvas.Zs, canvas.Ψ).PSI_interpolant
end

@recipe function plot_canvas(canvas::Canvas)
    Rs, Zs, Ψ, coils, Ψbnd = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.coils, canvas.Ψbnd

    aspect_ratio --> :equal
    pmin, pmax = extrema(Ψ)
    cmap = :diverging
    pext = max(abs(pmin), abs(pmax))
    clims --> (-pext, pext)
    @series begin
        seriestype --> :heatmap
        c --> cmap
        Rs, Zs, Ψ'
    end
    @series begin
        label --> nothing
        coils
    end
    @series begin
        seriestype --> :contour
        colorbar_entry --> false
        linewidth --> 2
        levels --> @SVector[Ψbnd]
        c --> :black
        Rs, Zs, Ψ'
    end
end