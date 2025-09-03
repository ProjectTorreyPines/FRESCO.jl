@kwdef mutable struct CoilState{T}
    flux::MVector{2,T} = zeros(MVector{2,Float64})
    current_per_turn::MVector{2,T} = zeros(MVector{2,Float64})
    voltage::T = zero(T)
    resistance::T = zero(T)
end

function CoilState(coil; initial_flux::T=0.0, voltage::T=0.0) where {T<:Real}
    flux = MVector{2,T}(initial_flux, zero(T))
    current_per_turn = MVector{2,T}(VacuumFields.current_per_turn(coil), zero(T))
    resistance = VacuumFields.resistance(coil)
    return CoilState(; flux, current_per_turn, voltage, resistance)
end


@kwdef mutable struct Canvas{T<:Real,DT<:Real,VC<:Vector{<:VacuumFields.AbstractCoil},II<:Interpolations.AbstractInterpolation,DI<:DataInterpolations.AbstractInterpolation,
    C1<:VacuumFields.AbstractCircuit,C2<:VacuumFields.AbstractCircuit}
    Rs::StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T},Int}
    Zs::StepRangeLen{T,Base.TwicePrecision{T},Base.TwicePrecision{T},Int}
    Ψ::Matrix{DT} = zeros(eltype(Rs), length(Rs), length(Zs))
    Ip::DT
    Fbnd::DT
    coils::VC
    Rw::Vector{T}
    Zw::Vector{T}
    Raxis::DT = zero(Ip)
    Zaxis::DT = zero(Ip)
    Ψaxis::DT = zero(Ip)
    Ψbnd::DT = zero(Ip)
    Rb_target::Vector{T}
    Zb_target::Vector{T}
    iso_cps::Vector{VacuumFields.IsoControlPoint{T}} = VacuumFields.IsoControlPoint{eltype(Rs)}[]
    flux_cps::Vector{VacuumFields.FluxControlPoint{T}} = VacuumFields.FluxControlPoint{eltype(Rs)}[]
    saddle_cps::Vector{VacuumFields.SaddleControlPoint{T}} = VacuumFields.SaddleControlPoint{eltype(Rs)}[]
    field_cps::Vector{VacuumFields.FieldControlPoint{T}} = VacuumFields.FieldControlPoint{eltype(Rs)}[]
    loop_cps::Vector{VacuumFields.IsoControlPoint{T}} = VacuumFields.IsoControlPoint{eltype(Rs)}[]
    surfaces::Vector{IMAS.SimpleSurface{DT}} = Vector{IMAS.SimpleSurface{eltype(Rs)}}(undef, length(Rs) - 1)
    Green_table::Array{T,3} = VacuumFields.Green_table(Rs, Zs, coils)
    fixed_coils::Vector{Int} = Int[]
    λ_regularize::T = 1e-14
    _Ψpl::Matrix{DT} = zero(Ψ)
    _Ψvac::Matrix{DT} = zero(Ψ)
    _Gbnd::Matrix{T} = compute_Gbnd(Rs, Zs)
    _U::Matrix{DT} = zero(Ψ)
    _Jt::Matrix{DT} = zero(Ψ)
    _Ψitp::II = ψ_interpolant(Rs, Zs, Ψ)
    _bnd::Vector{SVector{2,DT}} = SVector{2,eltype(Rs)}[]
    _rextrema::Tuple{DT,DT} = (zero(Ip), zero(Ip))
    _zextrema::Tuple{DT,DT} = (zero(Ip), zero(Ip))
    _is_inside::Matrix{Bool} = Matrix{Bool}(undef, size(Ψ))
    _is_in_wall::Matrix{Bool} = default_is_in_wall(Rs, Zs, Rw, Zw)
    _vs_circuit::C1 = default_vs_circuit(Rs, Zs)
    _rs_circuit::C2 = default_rs_circuit(Rs, Zs)
    _Ψ_at_coils::Vector{DT} = zeros(eltype(Rs), length(coils))
    _tmp_Ncoils::Vector{DT} = zeros(eltype(Rs), length(coils))
    _mutuals::Matrix{T} = Matrix{eltype(Rs)}(LinearAlgebra.I, length(coils), length(coils))
    _mutuals_LU::LU{T,Matrix{T},Vector{Int}} = lu(_mutuals)
    _a::Vector{T} = 1.0 ./ (1.0 .+ Base.step(Rs) ./ (2 .* Rs))
    _c::Vector{T} = 1.0 ./ (1.0 .- Base.step(Rs) ./ (2 .* Rs))
    _b::Vector{T} = _a .+ _c
    _MST::Matrix{T} = default_MST(Zs)
    _u::Matrix{DT} = zero(Ψ)
    _A::Vector{DT} = zero(Rs)
    _B::Vector{DT} = zero(Rs)
    _M::Tridiagonal{T,Vector{T}} = default_M(Rs)
    _LU::LU{T,Tridiagonal{T,Vector{T}},Vector{Int}} = default_LU(Rs)
    _S::Matrix{DT} = zero(Ψ)
    _tmp_Ψ::Matrix{DT} = zero(Ψ)
    _Vp::Vector{DT} = zeros(eltype(Rs), length(surfaces))
    _Vp_itp::DI = default_itp(surfaces)
    _gm1::Vector{DT} = zeros(eltype(Rs), length(surfaces))
    _gm1_itp::DI = default_itp(surfaces)
    _gm2p::Vector{DT} = zeros(eltype(Rs), length(surfaces))
    _gm2p_itp::DI = default_itp(surfaces)
    _gm9::Vector{DT} = zeros(eltype(Rs), length(surfaces))
    _gm9_itp::DI = default_itp(surfaces)
    _Fpol::Vector{DT} = zeros(eltype(Rs), length(surfaces))
    _Fpol_itp::DI = default_itp(surfaces)
    _rho::Vector{DT} = zeros(eltype(Rs), length(surfaces))
    _rho_itp::DI = default_itp(surfaces)
    _area::Vector{DT} = zeros(eltype(Rs), length(surfaces))
    _area_itp::DI = default_itp(surfaces)
    _r_cache::Vector{DT} = IMASutils.contour_cache(Ψ; aggression_level=3)[1]
    _z_cache::Vector{DT} = IMASutils.contour_cache(Ψ; aggression_level=3)[2]
end

function default_vs_circuit(Rs, Zs)
    R0, Z0 = 0.5 * (Rs[end] + Rs[1]), 0.5 * (Zs[end] + Zs[1])
    dZ = 0.5 * (Zs[end] - Zs[1])
    fout = 2.0
    vs_coils = [VacuumFields.PointCoil(R0, Z0 + fout * dZ), VacuumFields.PointCoil(R0, Z0 - fout * dZ)]
    return VacuumFields.SeriesCircuit(vs_coils, 0.0, [1, -1])
end

function default_rs_circuit(Rs, Zs)
    R0, Z0 = 0.5 * (Rs[end] + Rs[1]), 0.5 * (Zs[end] + Zs[1])
    dR, dZ = 0.5 * (Rs[end] - Rs[1]), 0.5 * (Zs[end] - Zs[1])
    fout = 2.0
    rs_coils = [VacuumFields.PointCoil(R0 + fout * dR, Z0 + dZ / 6), VacuumFields.PointCoil(R0 + fout * dR, Z0 - dZ / 6)]
    return VacuumFields.SeriesCircuit(rs_coils, 0.0, [1, 1])
end

function default_MST(Zs::AbstractVector{T}) where {T<:Real}
    return T[sqrt(2 / (length(Zs) - 1)) * sin(π * j * k / (length(Zs) - 1)) for j in 0:(length(Zs)-1), k in 0:(length(Zs)-1)]
end

function default_M(Rs::AbstractVector{T}) where {T<:Real}
    Nr = length(Rs)
    return Tridiagonal(zeros(T, Nr - 1), zero(Rs), zeros(T, Nr - 1))
end

function default_LU(Rs::AbstractVector{T}) where {T<:Real}
    Nr = length(Rs)
    return lu(Tridiagonal(zeros(T, Nr - 1), ones(T, Nr), zeros(T, Nr - 1)))
end

function default_is_in_wall(Rs, Zs, Rw, Zw)
    Wpts = collect(zip(Rw, Zw))
    return [(FRESCO.inpolygon((r, z), Wpts) == 1) for r in Rs, z in Zs]
end

function default_itp(surfaces::Vector{IMAS.SimpleSurface{T}}) where {T<:Real}
    x = range(zero(T), one(T), length(surfaces))
    return DataInterpolations.CubicSpline(zero(x), x; extrapolation=ExtrapolationType.None)
end


#==============================================================================
 Current Profiles
===============================================================================#
abstract type AbstractCurrentProfile end


mutable struct BetapIp{T} <: AbstractCurrentProfile
    betap::T
    alpha_m::T
    alpha_n::T
    Beta0::T
    L::T
end


mutable struct PaxisIp{T} <: AbstractCurrentProfile
    paxis::T
    alpha_m::T
    alpha_n::T
    L::T
    Beta0::T
end


mutable struct PprimeFFprime{F1<:DataInterpolations.AbstractInterpolation,F2<:DataInterpolations.AbstractInterpolation,T<:Real} <: AbstractCurrentProfile
    pprime::F1
    ffprime::F2
    ffp_scale::T
    grid::Symbol
end

PprimeFFprime(pprime::DataInterpolations.AbstractInterpolation, ffprime::DataInterpolations.AbstractInterpolation, grid::Symbol=:psi_norm) =
    PprimeFFprime(pprime, ffprime, 1.0, grid)

function PprimeFFprime(dd::IMAS.dd; grid::Symbol=:psi_norm)
    eqt1d = dd.equilibrium.time_slice[].profiles_1d
    check_grid(grid)
    x = getproperty(eqt1d, grid)
    pprime = DataInterpolations.CubicSpline(eqt1d.dpressure_dpsi, x; extrapolation=ExtrapolationType.Extension)
    ffprime = DataInterpolations.CubicSpline(eqt1d.f_df_dpsi, x; extrapolation=ExtrapolationType.Extension)
    return PprimeFFprime(pprime, ffprime, grid)
end


mutable struct PressureJtoR{F1<:DataInterpolations.AbstractInterpolation,F2<:DataInterpolations.AbstractInterpolation,T<:Real} <: AbstractCurrentProfile
    pressure::F1
    JtoR::F2 # <Jt / R>
    J_scale::T
    grid::Symbol
end

PressureJtoR(pressure::DataInterpolations.AbstractInterpolation, JtoR::DataInterpolations.AbstractInterpolation, grid::Symbol=:psi_norm) = PressureJtoR(pressure, JtoR, 1.0, grid)

function PressureJtoR(dd::IMAS.dd; j_p_from::Symbol=:equilibrium, grid::Symbol=:psi_norm)
    @assert j_p_from in (:equilibrium, :core_profiles)
    check_grid(grid)
    eqt1d = dd.equilibrium.time_slice[].profiles_1d
    if j_p_from === :equilibrium
        x = getproperty(eqt1d, grid)
        pressure = DataInterpolations.CubicSpline(eqt1d.pressure, x; extrapolation=ExtrapolationType.Extension)
        JtoR = DataInterpolations.CubicSpline(eqt1d.j_tor .* eqt1d.gm9, x; extrapolation=ExtrapolationType.Extension)
    else
        cp1d = dd.core_profiles.profiles_1d[]
        x = getproperty(cp1d.grid, grid)
        gm9 = IMAS.interp1d(getproperty(eqt1d, grid), eqt1d.gm9).(x)
        pressure = DataInterpolations.CubicSpline(cp1d.pressure, x; extrapolation=ExtrapolationType.Extension)
        JtoR = DataInterpolations.CubicSpline(cp1d.j_tor .* gm9, x; extrapolation=ExtrapolationType.Extension)
    end
    return PressureJtoR(pressure, JtoR, grid)
end


mutable struct PressureJt{F1<:DataInterpolations.AbstractInterpolation,F2<:DataInterpolations.AbstractInterpolation,T<:Real} <: AbstractCurrentProfile
    pressure::F1
    Jt::F2 # <Jt / R> / <1 / R>
    J_scale::T
    grid::Symbol
end

PressureJt(pressure::DataInterpolations.AbstractInterpolation, Jt::DataInterpolations.AbstractInterpolation, grid::Symbol=:psi_norm) = PressureJt(pressure, Jt, 1.0, grid)

function PressureJt(dd::IMAS.dd; j_p_from::Symbol=:equilibrium, grid::Symbol=:psi_norm)
    @assert j_p_from in (:equilibrium, :core_profiles)
    check_grid(grid)
    if j_p_from === :equilibrium
        eqt1d = dd.equilibrium.time_slice[].profiles_1d
        x = getproperty(eqt1d, grid)
        pressure = DataInterpolations.CubicSpline(eqt1d.pressure, x; extrapolation=ExtrapolationType.Extension)
        Jt = DataInterpolations.CubicSpline(eqt1d.j_tor, x; extrapolation=ExtrapolationType.Extension)
    else
        cp1d = dd.core_profiles.profiles_1d[]
        x = getproperty(cp1d.grid, grid)
        pressure = DataInterpolations.CubicSpline(cp1d.pressure, x; extrapolation=ExtrapolationType.Extension)
        Jt = DataInterpolations.CubicSpline(cp1d.j_tor, x; extrapolation=ExtrapolationType.Extension)
    end
    return PressureJt(pressure, Jt, grid)
end


mutable struct SigmaQ{F1<:DataInterpolations.AbstractInterpolation,F2<:DataInterpolations.AbstractInterpolation} <: AbstractCurrentProfile
    sigma::F1
    q::F2
    grid::Symbol
end