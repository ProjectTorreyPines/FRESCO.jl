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
    _Gbnd::Matrix{T}
    _U::Matrix{T}
    _Jt::Matrix{T}
    _Ψitp::I
    _bnd::Vector{SVector{2, T}}
    _rextrema::Tuple{T,T}
    _zextrema::Tuple{T,T}
    _is_inside::Matrix{Bool}
    _is_in_wall::Matrix{Bool}
    _Rb_target::Vector{T}
    _Zb_target::Vector{T}
    _iso_cps::Vector{VacuumFields.IsoControlPoint{T}}
    _flux_cps::Vector{VacuumFields.FluxControlPoint{T}}
    _saddle_cps::Vector{VacuumFields.SaddleControlPoint{T}}
    _vs_circuit::C1
    _rs_circuit::C2
    _Ψ_at_coils::Vector{T}
    _tmp_Ncoils::Vector{T}
    _fixed_coils::Vector{Int}
    _mutuals::Matrix{T}
    _mutuals_LU::LU{T, Matrix{T}, Vector{Int}}
    _a::Vector{T}
    _b::Vector{T}
    _c::Vector{T}
    _MST::Matrix{T}
    _u::Matrix{T}
    _A::Vector{T}
    _B::Vector{T}
    _M::Tridiagonal{T, Vector{T}}
    _LU::LU{T, Tridiagonal{T, Vector{T}}, Vector{Int}}
    _S::Matrix{T}
    _tmp_Ψ::Matrix{T}
    _surfaces::Vector{IMAS.SimpleSurface{T}}
    _Vp::Vector{T}
    _gm1::Vector{T}
    _gm9::Vector{T}
    _r_cache::Vector{T}
    _z_cache::Vector{T}
end

function Canvas(dd::IMAS.dd{T}, Nr::Int, Nz::Int=Nr;
                load_pf_active::Bool=true, load_pf_passive::Bool=true,
                Green_table::Array{T, 3}=T[;;;]) where {T<:Real}
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

    canvas = Canvas(dd, Rs, Zs, Ψ; load_pf_active, load_pf_passive, Green_table)

    update_bounds!(canvas)
    trace_surfaces!(canvas)
    gridded_Jtor!(canvas)
    return canvas
end

function Canvas(dd::IMAS.dd{T}, Rs::StepRangeLen, Zs::StepRangeLen,
                Ψ::Matrix{T}=zeros(T, length(Rs), length(Zs));
                coils=nothing, load_pf_active=true, load_pf_passive=true,
                x_points_weight::Float64=1.0, strike_points_weight::Float64=1.0,
                Green_table::Array{T, 3}=T[;;;]) where {T<:Real}

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

    # define current
    Ip = eqt.global_quantities.ip

    Nsurfaces = !ismissing(eqt.profiles_1d, :psi) ? length(eqt.profiles_1d.psi) : 129
    surfaces = Vector{IMAS.SimpleSurface{T}}(undef, Nsurfaces)

    canvas = Canvas(Rs, Zs, Ψ, Ip, coils, wall_r, wall_z, collect(boundary.r), collect(boundary.z), iso_cps, flux_cps, saddle_cps, surfaces; fixed_coils, Green_table)

    set_Ψvac!(canvas)
    canvas._Ψpl .= canvas.Ψ - canvas._Ψvac

    return canvas
end

function Canvas(Rs::AbstractRange{T},
                Zs::AbstractRange{T},
                Ip::T,
                coils::CoilVectorType,
                Rw::Vector{T},
                Zw::Vector{T},
                Rb_target::Vector{T},
                Zb_target::Vector{T},
                iso_cps::Vector{VacuumFields.IsoControlPoint{T}},
                flux_cps::Vector{VacuumFields.FluxControlPoint{T}},
                saddle_cps::Vector{VacuumFields.SaddleControlPoint{T}};
                fixed_coils::Vector{Int}=Int[],
                Green_table::Array{T, 3}=T[;;;]) where {T<:Real}
    Nr, Nz = length(Rs), length(Zs)
    Ψ = zeros(T, Nr, Nz)
    surfaces = Vector{IMAS.SimpleSurface{T}}(undef, Nr - 1)
    return Canvas(Rs, Zs, Ψ, Ip, coils, Rw, Zw,  Rb_target, Zb_target, iso_cps, flux_cps, saddle_cps, surfaces; fixed_coils, Green_table)
end

function Canvas(Rs::AbstractRange{T},
                Zs::AbstractRange{T},
                Ψ::Matrix{T},
                Ip::T,
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
                Green_table::Array{T, 3}=T[;;;]) where {T<:Real}
    Nr, Nz = length(Rs), length(Zs)
    @assert size(Ψ) == (Nr, Nz)
    hr = Base.step(Rs)

    R0, Z0 = 0.5 * (Rs[end] + Rs[1]), 0.5 * (Zs[end] + Zs[1])
    dR, dZ = 0.5 * (Rs[end] - Rs[1]), 0.5 * (Zs[end] - Zs[1])

    fout = 2.0
    vs_coils = [VacuumFields.PointCoil(R0, Z0 + fout * dZ), VacuumFields.PointCoil(R0, Z0 - fout * dZ)]
    vs_circuit = VacuumFields.SeriesCircuit(vs_coils, 0.0, [1, -1])
    rs_coils = [VacuumFields.PointCoil(R0 + fout * dR, Z0 + dZ / 6), VacuumFields.PointCoil(R0 + fout * dR, Z0 - dZ / 6)]
    rs_circuit = VacuumFields.SeriesCircuit(rs_coils, 0.0, [1, 1])

    Nc = length(coils)
    Ψ_at_coils = zeros(Nc)
    tmp_Ncoils = zero(Ψ_at_coils)
    mutuals = zeros(Nc, Nc) + LinearAlgebra.I
    mutuals_LU = lu(mutuals)

    a = @. (1.0 + hr / (2Rs)) ^ -1
    c = @. (1.0 - hr / (2Rs)) ^ -1
    b = a + c
    Ψpl = zero(Ψ)
    Ψvac = zero(Ψ)
    if isempty(Green_table)
        Gvac = Array{T, 3}(undef, Nr, Nz, Nc)
        for (k, coil) in enumerate(coils)
            # Green isn't threadsafe if coil is a MultiCoil and it's current is 0.0
            # so we set it to 1.0, compute the Green's functions, and then reset it
            if coil isa VacuumFields.MultiCoil
                Ic = VacuumFields.current(coil)
                Ic == 0.0 && VacuumFields.set_current!(coil, 1.0)
            end
            @inbounds @fastmath Threads.@threads for j in eachindex(Zs)
                z = Zs[j]
                for (i, r) in enumerate(Rs)
                    Gvac[i, j, k] = VacuumFields.Green(coil, r, z)
                end
            end
            if coil isa VacuumFields.MultiCoil
                Ic == 0.0 && VacuumFields.set_current!(coil, Ic)
            end
        end
    else
        @assert size(Green_table) == (Nr, Nz, Nc) "Green_table is incorrect size for grid and coils"
        Gvac = Green_table
    end
    Gbnd = compute_Gbnd(Rs, Zs)
    U = zero(Ψ)
    Jt = zero(Ψ)
    Ψitp = ψ_interpolant(Rs, Zs, Ψ)
    is_inside = Matrix{Bool}(undef, size(Ψ))
    Wpts = collect(zip(Rw, Zw))
    is_in_wall = [(FRESCO.inpolygon((r, z), Wpts) == 1) for r in Rs, z in Zs]
    u = zero(Ψ)
    A = zero(Rs)
    B = zero(Rs)
    MST = [sqrt(2 / (Nz - 1)) * sin(π * j * k / (Nz - 1)) for j in 0:(Nz-1), k in 0:(Nz-1)]
    M = Tridiagonal(zeros(T, Nr-1), ones(T, Nr), zeros(T, Nr-1))  # fill with ones so I can allocate lu Array
    LU = lu(M)
    M .= 0.0 # reset
    S = zero(Ψ)
    tmp_Ψ = zero(Ψ)
    zt = zero(T)
    Vp  = zeros(length(surfaces))
    gm1 = zeros(length(surfaces))
    gm9 = zeros(length(surfaces))
    r_cache, z_cache = IMASutils.contour_cache(Ψ; aggression_level=3)
    return Canvas(Rs, Zs, Ψ, Ip, coils, Rw, Zw, zt, zt, zt, zt, Ψpl, Ψvac, Gvac, Gbnd, U, Jt, Ψitp,
                  SVector{2,T}[], (0.0, 0.0), (0.0, 0.0), is_inside, is_in_wall, Rb_target, Zb_target,
                  iso_cps, flux_cps, saddle_cps,
                  vs_circuit, rs_circuit, Ψ_at_coils, tmp_Ncoils, fixed_coils, mutuals, mutuals_LU, a, b, c, MST, u,
                  A, B, M, LU, S, tmp_Ψ, surfaces, Vp, gm1, gm9, r_cache, z_cache)
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
                G[l, k] = Green(X[l], Y[l], X[k], Y[k])
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
                canvas._Rb_target, canvas._Zb_target
            end
        end
        if plot_control_points
            @series begin
                canvas._iso_cps
            end
            @series begin
                canvas._flux_cps
            end
            @series begin
                canvas._saddle_cps
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