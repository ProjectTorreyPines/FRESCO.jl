const CoilVectorType = AbstractVector{<:Union{VacuumFields.AbstractCoil, IMAS.pf_active__coil, IMAS.pf_active__coil___element}}

mutable struct Canvas{T<:Real, VC<:CoilVectorType, I<:Interpolations.AbstractInterpolation, C1<:VacuumFields.AbstractCircuit, C2<:VacuumFields.AbstractCircuit}
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
    _vs_circuit::C1
    _rs_circuit::C2
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

function Canvas(dd::IMAS.dd, Nr::Int, Nz::Int=Nr)
    eqt = dd.equilibrium.time_slice[]

    wall_r, wall_z = IMAS.first_wall(dd.wall)
    wall_r, wall_z = collect(wall_r), collect(wall_z)
    Rs, Zs = range(extrema(wall_r)..., Nr), range(extrema(wall_z)..., Nz)
    
    boundary = IMAS.closed_polygon(eqt.boundary.outline.r, eqt.boundary.outline.z)

    # define current
    Ip = eqt.global_quantities.ip

    # define coils
    coils = VacuumFields.MultiCoils(dd)

    return Canvas(Rs, Zs, Ip, coils, wall_r, wall_z, collect(boundary.r), collect(boundary.z))
end

function Canvas(Rs::AbstractRange{T},
                Zs::AbstractRange{T},
                Ip::T,
                coils::CoilVectorType,
                Rw::Vector{T},
                Zw::Vector{T},
                Rb_target::Vector{T},
                Zb_target::Vector{T}) where {T <: Real}
    Nr, Nz = length(Rs) - 1, length(Zs) - 1
    hr = (Rs[end] - Rs[1]) / Nr

    R0, Z0 = 0.5 * (Rs[end] + Rs[1]), 0.5 * (Zs[end] + Zs[1])
    dR, dZ = 0.5 * (Rs[end] - Rs[1]), 0.5 * (Zs[end] - Zs[1])

    fout = 2.0
    vs_coils = [VacuumFields.PointCoil(R0, Z0 + fout * dZ), VacuumFields.PointCoil(R0, Z0 - fout * dZ)]
    vs_circuit = VacuumFields.SeriesCircuit(vs_coils, 0.0, [1, -1])
    rs_coils = [VacuumFields.PointCoil(R0 + fout * dR, Z0 + dZ / 6), VacuumFields.PointCoil(R0 + fout * dR, Z0 - dZ / 6)]
    rs_circuit = VacuumFields.SeriesCircuit(rs_coils, 0.0, [1, 1])

    a = @. (1.0 + hr / (2Rs)) ^ -1
    c = @. (1.0 - hr / (2Rs)) ^ -1
    b = a + c
    Ψ = zeros(T, Nr + 1, Nz + 1)
    Ψpl = zero(Ψ)
    Ψvac = zero(Ψ)
    Gvac = [VacuumFields.Green(coil, r, z) for r in Rs, z in Zs, coil in coils]
    U = zero(Ψ)
    Jt = zero(Ψ)
    Ψitp = ψ_interpolant(Rs, Zs, Ψ)
    is_inside = Matrix{Bool}(undef, size(Ψ))
    u = zero(Ψ)
    A = zero(Rs)
    B = zero(Rs)
    MST = [sqrt(2 / Nz) * sin(π * j * k / Nz) for j in 0:Nz, k in 0:Nz]
    M = Tridiagonal(zeros(T, Nr), zeros(T, Nr+1), zeros(T, Nr))
    S = zero(Ψ)
    zt = zero(T)
    return Canvas(Rs, Zs, Ψ, Ip, coils, Rw, Zw, zt, zt, zt, zt, Ψpl, Ψvac, Gvac, U, Jt, Ψitp,
                  SVector{2,T}[], (0.0, 0.0), (0.0, 0.0), is_inside, Rb_target, Zb_target,
                  vs_circuit, rs_circuit, a, b, c, MST, u, A, B, M, S)
end

ψ_interpolant(r, z, psi) = Interpolations.cubic_spline_interpolation((r, z), psi; extrapolation_bc=Interpolations.Line())

function update_interpolation!(canvas::Canvas)
    canvas._Ψitp = ψ_interpolant(canvas.Rs, canvas.Zs, canvas.Ψ)
end

@recipe function plot_canvas(canvas::Canvas)
    Rs, Zs, Ψ, coils, Rw, Zw, Ψbnd, Rbt, Zbt = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.coils, canvas.Rw, canvas.Zw, canvas.Ψbnd, canvas._Rb_target, canvas._Zb_target

    aspect_ratio --> :equal
    pmin, pmax = extrema(Ψ)
    cmap = :diverging
    pext = max(abs(pmin - Ψbnd), abs(pmax - Ψbnd))
    clims --> (-pext + Ψbnd, pext + Ψbnd)
    xlabel --> "R (m)"
    ylabel --> "Z (m)"
    title --> "Poloidal Flux"
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
    # paxis = eq1d.pressure[1]
    # profile = PaxisIp(paxis, alpha_m, alpha_n)
    psi_norm = eq1d.psi_norm
    gpp = IMAS.interp1d(psi_norm, eq1d.dpressure_dpsi, :cubic)
    pprime  = x -> gpp(x)
    gffp =  IMAS.interp1d(psi_norm, eq1d.f_df_dpsi, :cubic)
    ffprime = x -> gffp(x)
    profile = PprimeFFprime(pprime, ffprime)
    canvas = Canvas(deepcopy(dd), Nr)
    return dd, profile, canvas
end