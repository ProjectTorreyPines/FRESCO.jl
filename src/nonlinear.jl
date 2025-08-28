dualize(::Type{S}, x::Number) where {S<:Dual} = convert(S, x)
dualize(::Type{S}, x::Union{AbstractArray, Tuple}) where {S<:Dual} = map(S, x)
dualize(::Type{S}, x::Vector{<:AbstractArray}) where {S<:Dual} = [dualize(S, xx) for xx in x]

function dualize(::Type{S}, surf::T) where {S<:Dual, T<:IMAS.SimpleSurface}
    args = [dualize(S, getproperty(surf, field)) for field in fieldnames(T)]
    return IMAS.SimpleSurface(args...)
end
function dualize(::Type{S}, surfaces::Vector{<:IMAS.SimpleSurface}) where {S<:Dual}
    return [dualize(S, surf) for surf in surfaces]
end

function dualize(::Type{S}, pitp::Interpolations.AbstractInterpolation) where {S<:Dual}
    Nr, Nz  = length.(Interpolations.axes(pitp))
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

function dualize(::Type{S}, coil::C) where {S<:Dual, C<:VacuumFields.AbstractSingleCoil}
    args = [(field === :current_per_turn) ? dualize(S, getproperty(coil, field)) : getproperty(coil, field) for field in fieldnames(C)]
    return C.name.wrapper(args...)
end
dualize(::Type{S}, coils::Vector{<:VacuumFields.AbstractCoil}) where {S<:Dual} = [dualize(S, coil) for coil in coils]
dualize(::Type{S}, mcoil::VacuumFields.MultiCoil) where {S<:Dual} = VacuumFields.MultiCoil(dualize(S, mcoil.coils), mcoil.orientation)
function dualize(::Type{S}, circuit::VacuumFields.SeriesCircuit) where {S<:Dual}
    return VacuumFields.SeriesCircuit(dualize(S, circuit.coils), dualize(S, circuit.current_per_turn), circuit.signs; copy_coils=false)
end

# USUSED
#dualize(::Type{S}, x::AbstractRange) where {S<:Dual} = range(S(first(x)); step=S(step(x)), length=length(x))
#dualize(::Type{S}, x::LU) where {S<:Dual} = LU(dualize(S, x.factors) x.ipiv, x.info)


const dualize_fields = (
    :Ψ, :Ip, :Fbnd, :coils, :Raxis, :Zaxis, :Ψaxis, :Ψbnd, :surfaces,
    :_Ψpl, :_Ψvac, :_U, :_Jt, :_Ψitp, :_bnd, :_rextrema, :_zextrema,
    :_vs_circuit, :_rs_circuit, :_Ψ_at_coils, :_tmp_Ncoils, :_u, :_A, :_B,
    :_S, :_tmp_Ψ, :_Vp, :_Vp_itp, :_gm1, :_gm1_itp, :_gm2p, :_gm2p_itp,
    :_gm9, :_gm9_itp, :_Fpol, :_Fpol_itp, :_rho, :_rho_itp, :_area, :_area_itp,
    :_r_cache, :_z_cache
)

function dualize(::Type{S}, canvas::C) where {S<:Dual, C<:Canvas}
    kwargs = Dict()
    for field in fieldnames(C)
        value = getproperty(canvas, field)
        kwargs[field] = (field in dualize_fields) ? dualize(S, value) : value
    end
    dual_canvas = Canvas(; kwargs...)
    update_interpolation!(dual_canvas)
    return dual_canvas
end