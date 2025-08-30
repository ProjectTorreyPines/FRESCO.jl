#==============================================================================
 Dualize: Functions to convert Canvas to use Dual numbers
==============================================================================#

const dualize_fields = (
    :Ψ, :Ip, :Fbnd, :coils, :Raxis, :Zaxis, :Ψaxis, :Ψbnd, :surfaces,
    :_Ψpl, :_Ψvac, :_U, :_Jt, :_Ψitp, :_bnd, :_rextrema, :_zextrema,
    :_vs_circuit, :_rs_circuit, :_Ψ_at_coils, :_tmp_Ncoils, :_u, :_A, :_B,
    :_S, :_tmp_Ψ, :_Vp, :_Vp_itp, :_gm1, :_gm1_itp, :_gm2p, :_gm2p_itp,
    :_gm9, :_gm9_itp, :_Fpol, :_Fpol_itp, :_rho, :_rho_itp, :_area, :_area_itp,
    :_r_cache, :_z_cache
)

function dualize(::Type{S}, canvas::C) where {S<:Dual,C<:Canvas}
    kwargs = Dict()
    for field in fieldnames(Canvas)
        value = getproperty(canvas, field)
        kwargs[field] = (field in dualize_fields) ? dualize(S, value) : value
    end
    dual_canvas = Canvas(; kwargs...)
    update_interpolation!(dual_canvas)
    return dual_canvas
end

dualize(::Type{S}, x::Number) where {S<:Dual} = convert(S, x)
dualize(::Type{S}, x::Union{AbstractArray,Tuple}) where {S<:Dual} = map(S, x)
dualize(::Type{S}, x::Vector{<:AbstractArray}) where {S<:Dual} = [dualize(S, xx) for xx in x]

function dualize(::Type{S}, surf::T) where {S<:Dual,T<:IMAS.SimpleSurface}
    args = [dualize(S, getproperty(surf, field)) for field in fieldnames(T)]
    return IMAS.SimpleSurface(args...)
end
function dualize(::Type{S}, surfaces::Vector{<:IMAS.SimpleSurface}) where {S<:Dual}
    return [dualize(S, surf) for surf in surfaces]
end

function dualize(::Type{S}, pitp::Interpolations.AbstractInterpolation) where {S<:Dual}
    Nr, Nz = length.(Interpolations.axes(pitp))
    rmin, rmax = Interpolations.bounds(pitp, 1)
    zmin, zmax = Interpolations.bounds(pitp, 2)
    r = range(rmin, rmax; length=Nr)
    z = range(zmin, zmax; length=Nz)
    psi = zeros(S, Nr, Nz)
    return ψ_interpolant(r, z, psi)
end

function dualize(::Type{S}, y_itp::DataInterpolations.CubicSpline) where {S<:Dual}
    x = y_itp.t
    y = dualize(S, y_itp.u)
    ext = y_itp.extrapolation_left
    @assert ext === y_itp.extrapolation_right
    return DataInterpolations.CubicSpline(y, x; extrapolation=ext)
end

function dualize(::Type{S}, coil::C) where {S<:Dual,C<:VacuumFields.AbstractSingleCoil}
    args = [(field === :current_per_turn) ? dualize(S, getproperty(coil, field)) : getproperty(coil, field) for field in fieldnames(C)]
    return C.name.wrapper(args...)
end
dualize(::Type{S}, coils::Vector{<:VacuumFields.AbstractCoil}) where {S<:Dual} = [dualize(S, coil) for coil in coils]
dualize(::Type{S}, mcoil::VacuumFields.MultiCoil) where {S<:Dual} = VacuumFields.MultiCoil(dualize(S, mcoil.coils), mcoil.orientation)
function dualize(::Type{S}, circuit::VacuumFields.SeriesCircuit) where {S<:Dual}
    return VacuumFields.SeriesCircuit(dualize(S, circuit.coils), dualize(S, circuit.current_per_turn), circuit.signs; copy_coils=false)
end

dualize(::Type{S}, profile::PprimeFFprime) where {S<:Dual} =
    PprimeFFprime(deepcopy(profile.pprime), deepcopy(profile.ffprime), dualize(S, profile.ffp_scale), profile.grid)
dualize(::Type{S}, profile::PressureJtoR) where {S<:Dual} =
    PressureJtoR(deepcopy(profile.pressure), deepcopy(profile.JtoR), dualize(S, profile.J_scale), profile.grid)
dualize(::Type{S}, profile::PressureJt) where {S<:Dual} =
    PressureJt(deepcopy(profile.pressure), deepcopy(profile.Jt), dualize(S, profile.J_scale), profile.grid)


#==============================================================================
 Sync Dual and regular canvases
==============================================================================#

function sync_dual_canvas!(
    dual_canvas::Canvas{D_T,D_DT,D_VC,D_II,D_DI,D_C1,D_C2},
    canvas::Canvas{T,DT,VC,II,DI,C1,C2}
) where {D_T,D_DT<:Dual,D_VC,D_II,D_DI,D_C1,D_C2,T,DT<:AbstractFloat,VC,II,DI,C1,C2}
    for field in dualize_fields
        sync_field!(dual_canvas, field, getproperty(canvas, field))
    end
    update_interpolation!(dual_canvas)
    return dual_canvas
end

function sync_field!(dual_canvas::Canvas{T,S,VC,II,DI,C1,C2}, field::Symbol, value::Union{Number,Tuple,DataInterpolations.AbstractInterpolation}) where {T,S<:Dual,VC,II,DI,C1,C2}
    return setproperty!(dual_canvas, field, dualize(S, value))
end

function sync_field!(dual_canvas::Canvas{T,S,VC,II,DI,C1,C2}, field::Symbol, value::AbstractArray) where {T,S<:Dual,VC,II,DI,C1,C2}
    dual_value = getproperty(dual_canvas, field)
    return dual_value .= value
end

function sync_field!(dual_canvas::Canvas{T,S,VC,II,DI,C1,C2}, field::Symbol, value::Vector{<:StaticArrays.SArray}) where {T,S<:Dual,VC,II,DI,C1,C2}
    dual_value = getproperty(dual_canvas, field)
    @assert length(dual_value) == length(value)
    for k in eachindex(value)
        dual_value[k] = dualize(S, value[k])
    end
end

function sync_field!(dual_canvas::Canvas{T,S,VC,II,DI,C1,C2}, field::Symbol, value::Vector{<:AbstractArray}) where {T,S<:Dual,VC,II,DI,C1,C2}
    dual_value = getproperty(dual_canvas, field)
    @assert length(dual_value) == length(value)
    for k in eachindex(value)
        dual_value[k] .= value[k]
    end
end

function sync_field!(dual_canvas::Canvas{T,S,VC,II,DI,C1,C2}, field::Symbol, value::Vector{SS}) where {T,S<:Dual,VC,II,DI,C1,C2,SS<:IMAS.SimpleSurface}
    dual_value = getproperty(dual_canvas, field)
    @assert length(dual_value) == length(value)
    for k in eachindex(value)
        dvk, vk = dual_value[k], value[k]
        dvk.psi = vk.psi
        dvk.r .= vk.r
        dvk.z .= vk.z
        dvk.ll .= vk.ll
        dvk.fluxexpansion .= vk.fluxexpansion
        dvk.int_fluxexpansion_dl = vk.int_fluxexpansion_dl
    end
end

# Interpolation will be updated separately, so no-op
sync_field!(dual_canvas::Canvas{T,S,VC,II,DI,C1,C2}, field::Symbol, value::Interpolations.AbstractInterpolation) where {T,S<:Dual,VC,II,DI,C1,C2} = return

function sync_field!(dual_canvas::Canvas{T,S,VC,II,DI,C1,C2}, field::Symbol, value::Vector{<:VacuumFields.AbstractCoil}) where {T,S<:Dual,VC,II,DI,C1,C2}
    dual_value = getproperty(dual_canvas, field)
    @assert length(dual_value) == length(value)
    for k in eachindex(value)
        dual_coil, coil = dual_value[k], value[k]
        @assert same_coil(dual_coil, coil)
        VacuumFields.set_current_per_turn!(dual_coil, VacuumFields.current_per_turn(coil))
    end
end

function same_coil(
    dual_coil::VacuumFields.MultiCoil{<:VacuumFields.AbstractSingleCoil{T1,S,T3,T4}},
    coil::VacuumFields.MultiCoil{<:VacuumFields.AbstractSingleCoil{T1,T,T3,T4}}
) where {T1,S<:Dual,T3,T4,T<:Real}
    if dual_coil.orientation != coil.orientation || length(dual_coil.coils) != length(coil.coils)
        return false
    end
    return all(same_coil(dual_coil.coils[k], coil.coils[k]) for k in eachindex(coil.coils))
end

function same_coil(dual_coil::VacuumFields.AbstractSingleCoil{T1,S,T3,T4}, coil::VacuumFields.AbstractSingleCoil{T1,T,T3,T4}) where {T1,S<:Dual,T3,T4,T<:Real}
    for field in fieldnames(typeof(dual_coil))
        if field !== :current_per_turn
            if getproperty(dual_coil, field) != getproperty(coil, field)
                return false
            end
        end
    end
    return true
end

function sync_field!(dual_canvas::Canvas{T,S,VC,II,DI,C1,C2}, field::Symbol, value::VacuumFields.SeriesCircuit) where {T,S<:Dual,VC,II,DI,C1,C2}
    dual_value = getproperty(dual_canvas, field)
    @assert dual_value.signs == value.signs
    @assert length(dual_value.coils) == length(value.coils)
    @assert all(same_coil(dual_value.coils[k], value.coils[k]) for k in eachindex(value.coils))
    return VacuumFields.update_coil_currents!(dual_value, value.current_per_turn)
end

#==============================================================================
 Residual
==============================================================================#

function residual(du::Matrix{DT}, Ψpl::Matrix{DT}, canvas::Canvas{T,S,VC,II,DI,C1,C2}, profile) where {DT<:Dual,T,S<:AbstractFloat,VC,II,DI,C1,C2}
    dual_canvas = dualize(DT, canvas)
    dual_profile = dualize(DT, profile)
    return residual(du, Ψpl, dual_canvas, dual_profile)
end

function residual(du::Matrix{S}, Ψpl::Matrix{S}, canvas::Canvas{T,S,VC,II,DI,C1,C2}, profile) where {T,S,VC,II,DI,C1,C2}
    canvas._Ψpl .= Ψpl
    # We can update the coil currents here if we need to via control
    FRESCO.set_Ψvac!(canvas)
    FRESCO.sync_Ψ!(canvas; update_vacuum=false, update_Ψitp=true)
    FRESCO.update_bounds!(canvas; update_Ψitp=false)

    # The current from the profiles
    FRESCO.Jtor!(canvas, profile; update_surfaces=true, compute_Ip_from=:fsa)
    FRESCO.invert_GS_zero_bnd!(canvas) # this defines U for the boundary integral
    FRESCO.set_boundary_flux!(canvas)
    FRESCO.invert_GS!(canvas; update_Ψitp=false)
    return du .= Ψpl .- canvas._Ψpl
end


#==============================================================================
 Utilities
==============================================================================#

function same_struct(x::T, y::T) where {T}
    ds = _diff(x, y, "canvas")
    if ds === nothing
        return true
    else
        ds
    end
end

function _diff(x::T, y::T, p::String) where {T}
    if isequal(x, y)
        return nothing
    elseif isprimitivetype(T) || T <: Number
        return p
    elseif hasmethod(getindex, Tuple{T,Int})
        length(x) != length(y) && return p
        for k in eachindex(x)
            dk = _diff(x[k], y[k], "$p[$k]")
            !isnothing(dk) && return dk
        end
        return nothing
    elseif isstructtype(T)
        for field in fieldnames(T)
            df = _diff(getproperty(x, field), getproperty(y, field), "$p.$field")
            !isnothing(df) && return df
        end
        return nothing
    end
    return "Failed"
end;