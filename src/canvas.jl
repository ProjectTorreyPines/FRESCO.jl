function Canvas(dd::IMAS.dd{T}, Nr::Int, Nz::Int=Nr; kwargs...) where {T<:Real}
    eqt = dd.equilibrium.time_slice[]
    eqt2d = IMAS.findfirst(:rectangular, eqt.profiles_2d)
    Rimas = IMAS.to_range(eqt2d.grid.dim1)
    Zimas = IMAS.to_range(eqt2d.grid.dim2)

    if (length(Rimas) == Nr) && (length(Zimas) == Nz)
        # use exact flux from dd
        Rs, Zs = Rimas, Zimas
    else
        Rs = range(Rimas[1], Rimas[end], Nr)
        Zs = range(Zimas[1], Zimas[end], Nz)
    end

    canvas = Canvas(dd, Rs, Zs; kwargs...)

    return canvas
end

function Canvas(dd::IMAS.dd{T}, Rs::StepRangeLen, Zs::StepRangeLen; kwargs...) where {T<:Real}
    eqt = dd.equilibrium.time_slice[]
    eqt2d = IMAS.findfirst(:rectangular, eqt.profiles_2d)
    if !isnothing(eqt2d)
        Rimas = IMAS.to_range(eqt2d.grid.dim1)
        Zimas = IMAS.to_range(eqt2d.grid.dim2)

        if (Rimas == Rs) && (Zimas == Zs)
            # use exact flux from dd
            Ψ = deepcopy(eqt2d.psi)
        else
            PSI_interpolant = ψ_interpolant(Rimas, Zimas, eqt2d.psi)
            Ψ = [PSI_interpolant(r, z) for r in Rs, z in Zs]
        end
    else
        Ψ = zeros(T, length(Rs), length(Zs))
    end

    return Canvas(dd, Rs, Zs, Ψ; kwargs...)
end

function Canvas(dd::IMAS.dd{T}, Rs::StepRangeLen, Zs::StepRangeLen, Ψ::Matrix{T};
                coils=nothing, load_pf_active=true, load_pf_passive=true,
                x_points_weight::Real=1.0, strike_points_weight::Real=1.0,
                active_x_points::AbstractVector{Int}=Int[],
                reference_flux_loop_index::Int=1,
                flux_loop_weights::AbstractVector{<:Real}=T[],
                magnetic_probe_weights::AbstractVector{<:Real}=T[],
                fixed_coils::Union{Nothing, Vector{Int}}=nothing,
                kwargs...) where {T<:Real}

    eqt = dd.equilibrium.time_slice[]

    wall_r, wall_z = map(collect, IMAS.first_wall(dd.wall))

    if !isempty(eqt.boundary)
        if isempty(wall_r)
            mxh = IMAS.MXH(eqt.boundary.outline.r, eqt.boundary.outline.z, 2)
            mxh.ϵ *= 1.1
            wall_r, wall_z = mxh(100)
        end
        iso_cps, saddle_cps = VacuumFields.boundary_control_points(dd; x_points_weight, strike_points_weight, active_x_points)
    else
        iso_cps = VacuumFields.IsoControlPoint{T}[]
        saddle_cps = VacuumFields.SaddleControlPoint{T}[]
    end

    if (!isempty(dd.magnetics) && !isempty(dd.magnetics.b_field_pol_probe) && !isempty(dd.magnetics.b_field_pol_probe[1].field) &&
        !isempty(dd.magnetics.flux_loop) && !isempty(dd.magnetics.flux_loop[1].flux))
        flux_cps, loop_cps, field_cps = VacuumFields.magnetic_control_points(dd; reference_flux_loop_index, flux_loop_weights, magnetic_probe_weights)
    else
        flux_cps = VacuumFields.FluxControlPoint{T}[]
        loop_cps = VacuumFields.IsoControlPoint{T}[]
        field_cps = VacuumFields.FieldControlPoint{T}[]
    end

    # define coils
    if coils === nothing
        coils = VacuumFields.MultiCoils(dd; load_pf_active, load_pf_passive)
        if fixed_coils === nothing
            fixed_coils = Int[]
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
    elseif fixed_coils === nothing
        fixed_coils = Int[]
    end

    # define current and F at boundary
    Ip = eqt.global_quantities.ip
    Fbnd = eqt.global_quantities.vacuum_toroidal_field.b0 * eqt.global_quantities.vacuum_toroidal_field.r0

    Nsurfaces = !ismissing(eqt.profiles_1d, :psi) ? length(eqt.profiles_1d.psi) : 129
    surfaces = Vector{IMAS.SimpleSurface{T}}(undef, Nsurfaces)

    if !isempty(eqt.boundary)
        boundary = IMAS.closed_polygon(eqt.boundary.outline.r, eqt.boundary.outline.z)
        Rb, Zb = collect(boundary.r), collect(boundary.z)
    else
        Rb, Zb = T[], T[]
    end
    canvas = Canvas(Rs, Zs, Ψ, Ip, Fbnd, coils, wall_r, wall_z, Rb,  Zb, iso_cps, flux_cps, saddle_cps, field_cps, loop_cps, surfaces; fixed_coils, kwargs...)


    set_Ψvac!(canvas)
    canvas._Ψpl .= canvas.Ψ - canvas._Ψvac

    if !all(iszero, Ψ)
        update_bounds!(canvas)
        trace_surfaces!(canvas)
        gridded_Jtor!(canvas)
    end

    return canvas
end

function Canvas(Rs::AbstractRange{T},
                Zs::AbstractRange{T},
                Ip::T,
                Fbnd::T,
                coils::Vector{<:VacuumFields.AbstractCoil},
                Rw::Vector{T},
                Zw::Vector{T},
                Rb_target::Vector{T},
                Zb_target::Vector{T},
                iso_cps::Vector{VacuumFields.IsoControlPoint{T}},
                flux_cps::Vector{VacuumFields.FluxControlPoint{T}},
                saddle_cps::Vector{VacuumFields.SaddleControlPoint{T}},
                field_cps::Vector{VacuumFields.FieldControlPoint{T}},
                loop_cps::Vector{VacuumFields.IsoControlPoint{T}};
                kwargs...) where {T<:Real}
    return Canvas(; Rs, Zs, Ip, coils, Rw, Zw, Fbnd, Rb_target, Zb_target, iso_cps, flux_cps, saddle_cps, field_cps, loop_cps, kwargs...)
end

function Canvas(Rs::AbstractRange{T},
                Zs::AbstractRange{T},
                Ψ::Matrix{T},
                Ip::T,
                Fbnd::T,
                coils::Vector{<:VacuumFields.AbstractCoil},
                Rw::Vector{T},
                Zw::Vector{T},
                Rb_target::Vector{T},
                Zb_target::Vector{T},
                iso_cps::Vector{VacuumFields.IsoControlPoint{T}},
                flux_cps::Vector{VacuumFields.FluxControlPoint{T}},
                saddle_cps::Vector{VacuumFields.SaddleControlPoint{T}},
                field_cps::Vector{VacuumFields.FieldControlPoint{T}},
                loop_cps::Vector{VacuumFields.IsoControlPoint{T}},
                surfaces::Vector{<:IMAS.SimpleSurface};
                fixed_coils::Vector{Int}=Int[],
                Green_table::Array{T, 3}=VacuumFields.Green_table(Rs, Zs, coils),
                kwargs...) where {T<:Real}
    Nr, Nz = length(Rs), length(Zs)
    @assert size(Ψ) == (Nr, Nz) "Ψ is incorrect size for Rs and Zs grid"
    @assert size(Green_table) == (Nr, Nz, length(coils)) "Green_table is incorrect size for grid and coils"

    return Canvas(; Rs, Zs, Ψ, Ip, Fbnd, coils, Rw, Zw, Rb_target, Zb_target, iso_cps, flux_cps, saddle_cps, field_cps, loop_cps, surfaces, fixed_coils, Green_table, kwargs...)
end

function Canvas(canvas0::C; kwargs...) where {C <: Canvas}
    # ── 1. current fields → NamedTuple ────────────────────────────────
    names = fieldnames(C)
    vals  = (getfield(canvas0, name) for name in names)
    nt    = NamedTuple{names}(vals)

    # ── 2. merge: kwargs override existing entries ───────────────────
    nt2 = (; nt..., kwargs...)

    # ── 3. build the new Canvas via the @kwdef constructor ───────────
    return Canvas(; nt2...)
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
            @series begin
                canvas.field_cps
            end
            @series begin
                canvas.loop_cps
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
