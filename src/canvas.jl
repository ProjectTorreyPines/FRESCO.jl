const CoilVectorType = AbstractVector{<:Union{VacuumFields.AbstractCoil, IMAS.pf_active__coil, IMAS.pf_active__coil___element}}

@kwdef mutable struct Canvas{T<:Real, VC<:CoilVectorType, II<:Interpolations.AbstractInterpolation, DI<:DataInterpolations.AbstractInterpolation,
                      C1<:VacuumFields.AbstractCircuit, C2<:VacuumFields.AbstractCircuit, Qs<:Union{Nothing, QED_system}}
    Rs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int}
    Zs::StepRangeLen{T, Base.TwicePrecision{T}, Base.TwicePrecision{T}, Int}
    Ψ::Matrix{T} = zeros(eltype(Rs), length(Rs), length(Zs))
    Ip::T
    Fbnd::T
    coils::VC
    Rw::Vector{T}
    Zw::Vector{T}
    Raxis::T = zero(Ip)
    Zaxis::T = zero(Ip)
    Ψaxis::T = zero(Ip)
    Ψbnd::T = zero(Ip)
    Rb_target::Vector{T}
    Zb_target::Vector{T}
    iso_cps::Vector{VacuumFields.IsoControlPoint{T}} = VacuumFields.IsoControlPoint{eltype(Rs)}[]
    flux_cps::Vector{VacuumFields.FluxControlPoint{T}} = VacuumFields.FluxControlPoint{eltype(Rs)}[]
    saddle_cps::Vector{VacuumFields.SaddleControlPoint{T}} = VacuumFields.SaddleControlPoint{eltype(Rs)}[]
    surfaces::Vector{IMAS.SimpleSurface{T}} = Vector{IMAS.SimpleSurface{eltype(Rs)}}(undef, length(Rs) - 1)
    Green_table::Array{T,3} = VacuumFields.Green_table(Rs, Zs, coils)
    fixed_coils::Vector{Int} = Int[]
    Qsystem::Qs = nothing
    _Ψpl::Matrix{T} = zero(Ψ)
    _Ψvac::Matrix{T} = zero(Ψ)
    _Gbnd::Matrix{T} = compute_Gbnd(Rs, Zs)
    _U::Matrix{T} = zero(Ψ)
    _Jt::Matrix{T} = zero(Ψ)
    _Ψitp::II = ψ_interpolant(Rs, Zs, Ψ)
    _bnd::Vector{SVector{2, T}} = SVector{2,eltype(Rs)}[]
    _rextrema::Tuple{T,T} = (zero(Ip), zero(Ip))
    _zextrema::Tuple{T,T} = (zero(Ip), zero(Ip))
    _is_inside::Matrix{Bool} = Matrix{Bool}(undef, size(Ψ))
    _is_in_wall::Matrix{Bool} = default_is_in_wall(Rs, Zs, Rw, Zw)
    _vs_circuit::C1 = default_vs_circuit(Rs, Zs)
    _rs_circuit::C2 = default_rs_circuit(Rs, Zs)
    _Ψ_at_coils::Vector{T} = zeros(eltype(Rs), length(coils))
    _tmp_Ncoils::Vector{T} = zeros(eltype(Rs), length(coils))
    _mutuals::Matrix{T} = Matrix{eltype(Rs)}(LinearAlgebra.I, length(coils), length(coils))
    _mutuals_LU::LU{T, Matrix{T}, Vector{Int}} = lu(_mutuals)
    _a::Vector{T} =  1.0 ./ (1.0 .+ Base.step(Rs) ./ (2 .* Rs))
    _c::Vector{T} =  1.0 ./ (1.0 .- Base.step(Rs) ./ (2 .* Rs))
    _b::Vector{T} = _a .+ _c
    _MST::Matrix{T} = default_MST(Zs)
    _u::Matrix{T} = zero(Ψ)
    _A::Vector{T} = zero(Rs)
    _B::Vector{T} = zero(Rs)
    _M::Tridiagonal{T, Vector{T}} = default_M(Rs)
    _LU::LU{T, Tridiagonal{T, Vector{T}}, Vector{Int}} = default_LU(Rs)
    _S::Matrix{T} = zero(Ψ)
    _tmp_Ψ::Matrix{T} = zero(Ψ)
    _Vp::Vector{T} = zeros(eltype(Rs), length(surfaces))
    _Vp_itp::DI = default_itp(surfaces)
    _gm1::Vector{T} = zeros(eltype(Rs), length(surfaces))
    _gm1_itp::DI = default_itp(surfaces)
    _gm2p::Vector{T} = zeros(eltype(Rs), length(surfaces))
    _gm2p_itp::DI = default_itp(surfaces)
    _gm9::Vector{T} = zeros(eltype(Rs), length(surfaces))
    _gm9_itp::DI = default_itp(surfaces)
    _Fpol::Vector{T} = zeros(eltype(Rs), length(surfaces))
    _Fpol_itp::DI = default_itp(surfaces)
    _rho::Vector{T} = zeros(eltype(Rs), length(surfaces))
    _rho_itp::DI = default_itp(surfaces)
    _area::Vector{T} = zeros(eltype(Rs), length(surfaces))
    _area_itp::DI = default_itp(surfaces)
    _r_cache::Vector{T} = IMASutils.contour_cache(Ψ; aggression_level=3)[1]
    _z_cache::Vector{T} = IMASutils.contour_cache(Ψ; aggression_level=3)[2]
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

function default_MST(Zs::AbstractVector{T}) where {T <: Real}
    T[sqrt(2 / (length(Zs) - 1)) * sin(π * j * k / (length(Zs) - 1)) for j in 0:(length(Zs)-1), k in 0:(length(Zs)-1)]
end

function default_M(Rs::AbstractVector{T}) where {T <: Real}
    Nr = length(Rs)
    return Tridiagonal(zeros(T, Nr-1), zero(Rs), zeros(T, Nr-1))
end

function default_LU(Rs::AbstractVector{T}) where {T <: Real}
    Nr = length(Rs)
    return lu(Tridiagonal(zeros(T, Nr-1), ones(T, Nr), zeros(T, Nr-1)))
end

function default_is_in_wall(Rs, Zs, Rw, Zw)
    Wpts = collect(zip(Rw, Zw))
    return [(FRESCO.inpolygon((r, z), Wpts) == 1) for r in Rs, z in Zs]
end

function default_itp(surfaces::Vector{IMAS.SimpleSurface{T}}) where {T <: Real}
    x = range(zero(T), one(T), length(surfaces))
    return DataInterpolations.CubicSpline(zero(x), x; extrapolation=ExtrapolationType.None)
end

function Canvas(dd::IMAS.dd{T}, Nr::Int, Nz::Int=Nr; kwargs...) where {T<:Real}
    eqt = dd.equilibrium.time_slice[]
    eqt2d = IMAS.findfirst(:rectangular, eqt.profiles_2d)
    Rimas = IMAS.to_range(eqt2d.grid.dim1)
    Zimas = IMAS.to_range(eqt2d.grid.dim2)

    if (length(Rimas) == Nr) && (length(Zimas) == Nz)
        # use exact flux from dd
        Rs, Zs, Ψ = Rimas, Zimas, deepcopy(eqt2d.psi)
    else
        Rs = range(Rimas[1], Rimas[end], Nr)
        Zs = range(Zimas[1], Zimas[end], Nz)
        PSI_interpolant = ψ_interpolant(Rimas, Zimas, eqt2d.psi)
        Ψ = [PSI_interpolant(r, z) for r in Rs, z in Zs]
    end

    canvas = Canvas(dd, Rs, Zs, Ψ; kwargs...)

    update_bounds!(canvas)
    trace_surfaces!(canvas)
    gridded_Jtor!(canvas)
    return canvas
end

function Canvas(dd::IMAS.dd{T}, Rs::StepRangeLen, Zs::StepRangeLen,
                Ψ::Matrix{T}=zeros(T, length(Rs), length(Zs));
                coils=nothing, load_pf_active=true, load_pf_passive=true,
                x_points_weight::Float64=1.0, strike_points_weight::Float64=1.0,
                active_x_points::AbstractVector{Int}=Int[], kwargs...) where {T<:Real}

    eqt = dd.equilibrium.time_slice[]
    boundary = IMAS.closed_polygon(eqt.boundary.outline.r, eqt.boundary.outline.z)

    wall_r, wall_z = map(collect, IMAS.first_wall(dd.wall))
    if isempty(wall_r)
        mxh = IMAS.MXH(eqt.boundary.outline.r, eqt.boundary.outline.z, 2)
        mxh.ϵ *= 1.1
        wall_r, wall_z = mxh(100)
    end

    # Boundary control points
    iso_cps = VacuumFields.IsoControlPoints(eqt.boundary.outline.r, eqt.boundary.outline.z)

    strike_weight = strike_points_weight / length(eqt.boundary.strike_point)
    strike_cps = VacuumFields.IsoControlPoint{T}[VacuumFields.IsoControlPoint{T}(strike_point.r, strike_point.z, iso_cps[1].R2, iso_cps[1].Z2, strike_weight) for strike_point in eqt.boundary.strike_point]
    append!(iso_cps, strike_cps)

    # Flux control points
    flux_cps = VacuumFields.FluxControlPoint{T}[]

    # Saddle control points
    saddle_weight = x_points_weight / length(eqt.boundary.x_point)
    saddle_cps = VacuumFields.SaddleControlPoint{T}[VacuumFields.SaddleControlPoint{T}(x_point.r, x_point.z, saddle_weight) for x_point in eqt.boundary.x_point]
    xiso_cps = Vector{VacuumFields.IsoControlPoint{T}}(undef, length(active_x_points))
    for (k, ax) in enumerate(active_x_points)
        xiso_cps[k] = VacuumFields.IsoControlPoint{T}(eqt.boundary.x_point[ax].r, eqt.boundary.x_point[ax].z, iso_cps[1].R2, iso_cps[1].Z2, saddle_weight)
    end
    append!(iso_cps, xiso_cps)

    # define coils
    fixed_coils = Int[]
    if coils === nothing
        coils = VacuumFields.MultiCoils(dd; load_pf_active, load_pf_passive)
        if load_pf_active
            kpassive0 = length(dd.pf_active.coil)
            for (k, coil) in enumerate(dd.pf_active.coil)
                if :shaping ∉ (IMAS.index_2_name(coil.function)[f.index] for f in coil.function)
                    push!(fixed_coils, k)
                end
            end
        else
            kpassive0 = 0
        end
        if load_pf_passive
            fixed_coils = vcat(fixed_coils, kpassive0 .+ eachindex(dd.pf_passive.loop))
        end
    end

    # define current and F at boundary
    Ip = eqt.global_quantities.ip
    Fbnd = eqt.global_quantities.vacuum_toroidal_field.b0 * eqt.global_quantities.vacuum_toroidal_field.r0

    Nsurfaces = !ismissing(eqt.profiles_1d, :psi) ? length(eqt.profiles_1d.psi) : 129
    surfaces = Vector{IMAS.SimpleSurface{T}}(undef, Nsurfaces)
    canvas = Canvas(Rs, Zs, Ψ, Ip, Fbnd, coils, wall_r, wall_z, collect(boundary.r), collect(boundary.z), iso_cps, flux_cps, saddle_cps, surfaces; fixed_coils, kwargs...)

    set_Ψvac!(canvas)
    canvas._Ψpl .= canvas.Ψ - canvas._Ψvac

    return canvas
end

function Canvas(Rs::AbstractRange{T},
                Zs::AbstractRange{T},
                Ip::T,
                Fbnd::T,
                coils::CoilVectorType,
                Rw::Vector{T},
                Zw::Vector{T},
                Rb_target::Vector{T},
                Zb_target::Vector{T},
                iso_cps::Vector{VacuumFields.IsoControlPoint{T}},
                flux_cps::Vector{VacuumFields.FluxControlPoint{T}},
                saddle_cps::Vector{VacuumFields.SaddleControlPoint{T}};
                kwargs...) where {T<:Real}
    return Canvas(; Rs, Zs, Ip, coils, Rw, Zw, Fbnd, Rb_target, Zb_target, iso_cps, flux_cps, saddle_cps, kwargs...)
end

function Canvas(Rs::AbstractRange{T},
                Zs::AbstractRange{T},
                Ψ::Matrix{T},
                Ip::T,
                Fbnd::T,
                coils::CoilVectorType,
                Rw::Vector{T},
                Zw::Vector{T},
                Rb_target::Vector{T},
                Zb_target::Vector{T},
                iso_cps::Vector{VacuumFields.IsoControlPoint{T}},
                flux_cps::Vector{VacuumFields.FluxControlPoint{T}},
                saddle_cps::Vector{VacuumFields.SaddleControlPoint{T}},
                surfaces::Vector{<:IMAS.SimpleSurface};
                fixed_coils::Vector{Int}=Int[],
                Green_table::Array{T, 3}=VacuumFields.Green_table(Rs, Zs, coils),
                Qsystem::Union{Nothing, QED_system}=nothing) where {T<:Real}
    Nr, Nz = length(Rs), length(Zs)
    @assert size(Ψ) == (Nr, Nz) "Ψ is incorrect size for Rs and Zs grid"
    @assert size(Green_table) == (Nr, Nz, length(coils)) "Green_table is incorrect size for grid and coils"

    return Canvas(; Rs, Zs, Ψ, Ip, Fbnd, coils, Rw, Zw, Rb_target, Zb_target, iso_cps, flux_cps, saddle_cps, surfaces, fixed_coils, Green_table, Qsystem)
end

function bnd2mat(Nr::Int, Nz::Int, k::Int)
    if k <= Nr
        #bottom
        return k, 1
    elseif k <= Nr + Nz - 1
        #right
        return Nr, k - Nr + 1
    elseif k <= 2Nr + Nz - 2
        # top (reversed)
        return 2Nr + Nz - k - 1, Nz
    else
        # left (reversed)
        return 1, 2Nz + 2Nr - k - 2
    end
end

Nbnd(Nr::Int, Nz::Int) = 2 * (Nr + Nz) - 4

function compute_Gbnd(Rs::AbstractRange{T}, Zs::AbstractRange{T}) where {T<:Real}
    Nr, Nz = length(Rs), length(Zs)
    Nb = Nbnd(Nr, Nz)
    X = zeros(T, Nb)
    Y = zeros(T, Nb)
    for k in eachindex(X)
        i, j = bnd2mat(Nr, Nz, k)
        X[k] = Rs[i]
        Y[k] = Zs[j]
    end

    G = Matrix{T}(undef, Nb, Nb)
    for k in 1:Nb
        for l in k:Nb
            if k == l
                G[k, l] = 0.0
            else
                G[l, k] = VacuumFields.Green(X[l], Y[l], X[k], Y[k])
                G[k, l] = G[l, k]
            end
        end
    end
    return G
end

const ITP = Interpolations
function ψ_interpolant(r, z, psi)
    return ITP.scale(ITP.interpolate(psi, ITP.BSpline(ITP.Cubic(ITP.Line(ITP.OnGrid())))), r, z)
    #Interpolations.cubic_spline_interpolation((r, z), psi; extrapolation_bc=Interpolations.Line())
end

# This gets into the weeds of how Interpolations works to eliminate some unnecessary allocation in prefilter()
# If it every breaks, we can go back to the first commented line
function update_interpolation!(canvas::Canvas)
    # BCL 11/20/24: Use this if it breaks
    # canvas._Ψitp = ψ_interpolant(canvas.Rs, canvas.Zs, canvas.Ψ)

    itp = canvas._Ψitp.itp
    A = canvas.Ψ
    T = eltype(A)
    coefs = itp.coefs
    fill!(coefs, zero(T))
    indsA = axes(A)
    Interpolations.ct!(coefs, indsA, A, indsA)
    Interpolations.prefilter!(T, coefs, itp.it)
    return canvas._Ψitp
end

@recipe function plot_canvas(canvas::Canvas; plot_coils=true, plot_target_boundary=false, plot_control_points=true)
    Rs, Zs, Ψ, Rw, Zw, Ψbnd = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Rw, canvas.Zw, canvas.Ψbnd

    aspect_ratio --> :equal
    pmin, pmax = extrema(Ψ)
    cmap = :diverging
    pext = max(abs(pmin - Ψbnd), abs(pmax - Ψbnd))
    clims --> (-pext + Ψbnd, pext + Ψbnd)
    xlabel --> "R (m)"
    ylabel --> "Z (m)"
    title --> "Poloidal Flux"
    if plot_coils !== :only
        @series begin
            seriestype --> :heatmap
            c --> cmap
            Rs, Zs, Ψ'
        end
    end
    if plot_coils != false
        @series begin
            label --> nothing
            if plot_coils === :only
                alpha --> nothing
                color_by --> :current
            elseif plot_coils == true
                color_by --> :current
                cname := :BrBG_7
                colorbar_entry := false
            end
            canvas.coils
        end
    end
    if plot_coils !== :only
        @series begin
            label --> nothing
            seriestype --> :path
            linewidth --> 2
            linestyle --> :solid
            c --> :black
            Rw, Zw
        end
        if plot_target_boundary
            @series begin
                label --> nothing
                seriestype --> :path
                linewidth --> 3
                linestyle --> :solid
                c --> :limegreen
                canvas.Rb_target, canvas.Zb_target
            end
        end
        if plot_control_points
            @series begin
                canvas.iso_cps
            end
            @series begin
                canvas.flux_cps
            end
            @series begin
                canvas.saddle_cps
            end
        end
        @series begin
            seriestype --> :contour
            colorbar_entry --> false
            linewidth --> 3
            levels --> @SVector[Ψbnd]
            c --> :black
            linestyle --> :dash
            Rs, Zs, Ψ'
        end
    end
end
