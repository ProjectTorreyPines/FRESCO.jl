const CoilVectorType = AbstractVector{<:Union{VacuumFields.AbstractCoil, IMAS.pf_active__coil, IMAS.pf_active__coil___element}}

mutable struct Canvas{T<:Real, VC<:CoilVectorType, I<:Interpolations.AbstractInterpolation}
    Rs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int}
    Zs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int}
    Ψ::Matrix{T}
    Ip::T
    coils::VC
    Rw::Vector{T}
    Zw::Vector{T}
    Raxis::T
    Zaxis::T
    Ψaxis::T
    Ψbnd::T
    _Ψpl::Matrix{T}
    _Ψvac::Matrix{T}
    _Gvac::Array{T,3}
    _U::Matrix{T}
    _Jt::Matrix{T}
    _Ψitp::I
    _bnd::Vector{SVector{2, T}}
    _rextrema::Tuple{T,T}
    _zextrema::Tuple{T,T}
    _is_inside::Matrix{Bool}
    _Rb_target::Vector{T}
    _Zb_target::Vector{T}
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
    Rw, Zw = layer[k].outline.r, layer[k].outline.z
    Rs, Zs = range(minimum(Rw), maximum(Rw), Nr), range(minimum(Zw), maximum(Zw), Nz)

    eqt = dd.equilibrium.time_slice[]
    Rb_target = eqt.boundary.outline.r
    Zb_target = eqt.boundary.outline.z
    if (Rb_target[1] ≈ Rb_target[end]) && (Zb_target[1] ≈ Zb_target[end])
        Rb_target = Rb_target[1:end-1]
        Zb_target = Zb_target[1:end-1]
    end

    # define current
    eqt = dd.equilibrium.time_slice[]
    Ip = eqt.global_quantities.ip

    # define coils
    coils = VacuumFields.MultiCoils(dd)

    return Canvas(Rs, Zs, Ip, coils, Rw, Zw, Rb_target, Zb_target)
end

function Canvas(Rs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int},
                Zs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int},
                Ip::T,
                coils::CoilVectorType,
                Rw::Vector{T}, Zw::Vector{T},
                Rb_target::Vector{T}, Zb_target::Vector{T}) where {T <: Real}
    Nr, Nz = length(Rs) - 1, length(Zs) - 1
    hr = (Rs[end] - Rs[1]) / Nr
    a = @. (1.0 + hr / (2Rs)) ^ -1
    c = @. (1.0 - hr / (2Rs)) ^ -1
    b = a + c
    Ψ = zeros(T, Nr + 1, Nz + 1)
    Ψpl = zero(Ψ)
    Ψvac = zero(Ψ)
    Gvac = [VacuumFields.Green(coil, r, z) for r in Rs, z in Zs, coil in coils]
    U = zero(Ψ)
    Jt = zero(Ψ)
    Ψitp = IMAS.ψ_interpolant(Rs, Zs, Ψ).PSI_interpolant
    is_inside = Matrix{Bool}(undef, size(Ψ))
    u = zero(Ψ)
    A = zero(Rs)
    B = zero(Rs)
    MST = [sqrt(2 / Nz) * sin(π * j * k / Nz) for j in 0:Nz, k in 0:Nz]
    M = Tridiagonal(zeros(T, Nr), zeros(T, Nr+1), zeros(T, Nr))
    S = zero(Ψ)
    zt = zero(T)
    return Canvas(Rs, Zs, Ψ, Ip, coils, Rw, Zw, zt, zt, zt, zt, Ψpl, Ψvac, Gvac, U, Jt, Ψitp,
                  SVector{2,T}[], (0.0, 0.0), (0.0, 0.0), is_inside, Rb_target, Zb_target, a, b, c,
                  MST, u, A, B, M, S)
end

function update_interpolation!(canvas::Canvas)
    canvas._Ψitp = IMAS.ψ_interpolant(canvas.Rs, canvas.Zs, canvas.Ψ).PSI_interpolant
end

@recipe function plot_canvas(canvas::Canvas)
    Rs, Zs, Ψ, coils, Rw, Zw, Ψbnd, Rbt, Zbt = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.coils, canvas.Rw, canvas.Zw, canvas.Ψbnd, canvas._Rb_target, canvas._Zb_target

    aspect_ratio --> :equal
    pmin, pmax = extrema(Ψ)
    cmap = :diverging
    pext = max(abs(pmin - Ψbnd), abs(pmax - Ψbnd))
    clims --> (-pext + Ψbnd, pext + Ψbnd)
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
        label --> nothing
        seriestype --> :path
        linewidth --> 2
        linestyle --> :solid
        c --> :black
        Rw, Zw
    end
    @series begin
        label --> nothing
        seriestype --> :path
        linewidth --> 2
        linestyle --> :solid
        c --> :gray
        Rbt, Zbt
    end
    @series begin
        seriestype --> :contour
        colorbar_entry --> false
        linewidth --> 2
        levels --> @SVector[Ψbnd]
        c --> :black
        linestyle --> :dash
        Rs, Zs, Ψ'
    end
end


# Psuedo-temporary initialization function
function init_from_dd(file::String=(@__DIR__) * "/../examples/D3D_case/dd.json";
                      alpha_m::Real = 0.6, alpha_n::Real = 0.6, Nr::Int=65, Nz::Int=Nr)
    dd = IMAS.json2imas(file)
    eq1d = dd.equilibrium.time_slice[].profiles_1d
    paxis = eq1d.pressure[1]
    profile = PaxisIp(paxis, alpha_m, alpha_n)
    canvas = Canvas(deepcopy(dd), Nr)
    return dd, profile, canvas
end