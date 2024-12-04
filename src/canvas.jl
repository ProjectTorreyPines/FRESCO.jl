const CoilVectorType = AbstractVector{<:Union{VacuumFields.AbstractCoil, IMAS.pf_active__coil, IMAS.pf_active__coil___element}}

mutable struct Canvas{T<:Real, VC<:CoilVectorType, I<:AbstractInterpolation, C1<:VacuumFields.AbstractCircuit, C2<:VacuumFields.AbstractCircuit}
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
    _vs_circuit::C1
    _rs_circuit::C2
    _Ψ_at_coils::Vector{T}
    _tmp_Ncoils::Vector{T}
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
    _r_cache::Vector{T}
    _z_cache::Vector{T}
end

function Canvas(dd::IMAS.dd{T}, Nr::Int, Nz::Int=Nr; load_pf_active=true, load_pf_passive=true) where {T<:Real}

    wall_r, wall_z = IMAS.first_wall(dd.wall)
    wall_r, wall_z = collect(wall_r), collect(wall_z)
    #Rs, Zs = range(extrema(wall_r)..., Nr), range(extrema(wall_z)..., Nz)

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

    boundary = IMAS.closed_polygon(eqt.boundary.outline.r, eqt.boundary.outline.z)

    # define current
    Ip = eqt.global_quantities.ip

    # define coils
    coils = VacuumFields.MultiCoils(dd; load_pf_active, load_pf_passive)

    Ns = length(eqt.profiles_1d.psi)
    surfaces = Vector{IMAS.SimpleSurface{T}}(undef, Ns)

    canvas = Canvas(Rs, Zs, Ψ, Ip, coils, wall_r, wall_z, collect(boundary.r), collect(boundary.z), surfaces)

    set_Ψvac!(canvas)
    canvas._Ψpl .= canvas.Ψ - canvas._Ψvac
    update_bounds!(canvas)
    trace_surfaces!(canvas)
    gridded_Jtor!(canvas)

    return canvas
end

function Canvas(Rs::AbstractRange{T},
                Zs::AbstractRange{T},
                Ip::T,
                coils::CoilVectorType,
                Rw::Vector{T},
                Zw::Vector{T},
                Rb_target::Vector{T},
                Zb_target::Vector{T}) where {T<:Real}
    Nr, Nz = length(Rs), length(Zs)
    Ψ = zeros(T, Nr, Nz)
    surfaces = Vector{IMAS.SimpleSurface{T}}(undef, Nr - 1)
    return Canvas(Rs, Zs, Ψ, Ip, coils, Rw, Zw, Rb_target, Zb_target, surfaces)
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
                surfaces::Vector{<:IMAS.SimpleSurface}) where {T<:Real}
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
    Gvac = [VacuumFields.Green(coil, r, z) for r in Rs, z in Zs, coil in coils]
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
    r_cache, z_cache = IMASutils.contour_cache(Ψ)
    return Canvas(Rs, Zs, Ψ, Ip, coils, Rw, Zw, zt, zt, zt, zt, Ψpl, Ψvac, Gvac, Gbnd, U, Jt, Ψitp,
                  SVector{2,T}[], (0.0, 0.0), (0.0, 0.0), is_inside, is_in_wall, Rb_target, Zb_target,
                  vs_circuit, rs_circuit, Ψ_at_coils, tmp_Ncoils, mutuals, mutuals_LU, a, b, c, MST, u,
                  A, B, M, LU, S, tmp_Ψ, surfaces, Vp, gm1, r_cache, z_cache)
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

@recipe function plot_canvas(canvas::Canvas; plot_coils=true)
    Rs, Zs, Ψ, coils, Rw, Zw, Ψbnd, Rbt, Zbt = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.coils, canvas.Rw, canvas.Zw, canvas.Ψbnd, canvas._Rb_target, canvas._Zb_target

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
            coils
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
        @series begin
            label --> nothing
            seriestype --> :path
            linewidth --> 3
            linestyle --> :solid
            c --> :limegreen
            Rbt, Zbt
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