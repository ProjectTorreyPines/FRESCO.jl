# Standard G-S solver
function solve!(canvas::Canvas, profile::AbstractCurrentProfile, Nout::Int, Nin::Int;
                Rtarget = 0.5 * sum(extrema(canvas.Rb_target)),
                Ztarget = canvas.Zb_target[argmax(canvas.Rb_target)],
                debug=0,
                relax::Real=0.5,
                tolerance::Real=0.0,
                control::Union{Nothing, Symbol}=:shape,
                fixed_coils::AbstractVector{Int}=canvas.fixed_coils,
                initialize_current::Bool=(all(iszero, canvas.Ψ) && control !== :eddy),
                initialize_mutuals=(control === :eddy),
                compute_Ip_from::Symbol=:fsa)

    @assert control in (nothing, :shape, :vertical, :radial, :position, :eddy)

    if initialize_current
        if control === :eddy
            @warn "Eddy control should be used from existing equilibrium solution, so initialize_current should likely be false."
        end
        J = (x,y) -> initial_current(canvas, x, y)
        gridded_Jtor!(canvas, J)
    else
        gridded_Jtor!(canvas)
    end

    set_Ψvac!(canvas)

    if initialize_mutuals
        set_mutuals!(canvas)
        set_flux_at_coils!(canvas)
    end

    if control === :shape
        coils, iso_cps, flux_cps, saddle_cps = canvas.coils, canvas.iso_cps, canvas.flux_cps, canvas.saddle_cps
        @views active_coils = isempty(fixed_coils) ? coils : coils[setdiff(eachindex(coils), fixed_coils)]
        Acps = VacuumFields.define_A(active_coils; flux_cps, saddle_cps, iso_cps)
        b_offset = zeros(size(Acps, 1))
        fcs = @views coils[fixed_coils]
        VacuumFields.offset_b!(b_offset; flux_cps, saddle_cps, iso_cps, fixed_coils=fcs)

        return _solve!(canvas, profile, Nout, Nin; debug, relax, tolerance, control, compute_Ip_from, initialize_current, fixed_coils, Acps, b_offset)

    elseif control in (:vertical, :radial, :position)
        return _solve!(canvas, profile, Nout, Nin; debug, relax, tolerance, control, compute_Ip_from, initialize_current, Rtarget, Ztarget)

    else
        return _solve!(canvas, profile, Nout, Nin; debug, relax, tolerance, control, compute_Ip_from, initialize_current)
    end
end


# Implicit, time-dependent solver
function implicit_solve!(canvas::Canvas, pressure::DataInterpolations.AbstractInterpolation, Nout::Int, Nin::Int, tmax::Real, Nt::Int;
                         debug=0,
                         relax::Real=0.5,
                         tolerance::Real=0.0,
                         V_waveforms::Vector{<:QED.Waveform}=canvas.Qsystem.Qbuild.V_waveforms,
                         preprocess_canvas::Bool=true,
                         compute_Ip_from::Symbol=:fsa)

    @assert !isnothing(canvas.Qsystem)

    # build JtoR profile from Qstate
    Qstate = canvas.Qsystem.Qstate
    x = Qstate.ρ
    JtoR = DataInterpolations.CubicSpline(QED.Jt_R(Qstate), x; extrapolation=ExtrapolationType.Extension)
    profile = PressureJtoR(pressure, JtoR, :rho_tor_norm)

    # Setup the Qsystem
    setup_Qsystem!(canvas, profile; tmax, Nt, V_waveforms, preprocess_canvas, compute_Ip_from)

    # get mutual matrix from Qbuild
    canvas._mutuals .= canvas.Qsystem.Qbuild.Mcc

    # initialize vacuum fields
    set_Ψvac!(canvas)

    return _solve!(canvas, profile, Nout, Nin; debug, relax, tolerance, control=:implicit, compute_Ip_from)
end


# common solve
function _solve!(canvas::Canvas, profile::AbstractCurrentProfile, Nout::Int, Nin::Int;
                 debug, relax::Real, tolerance::Real, control::Union{Nothing, Symbol}, compute_Ip_from::Symbol,
                 initialize_current::Bool=false, Rtarget=nothing, Ztarget=nothing, fixed_coils=nothing, Acps=nothing, b_offset=nothing)

    Ψ, Ψpl = canvas.Ψ, canvas._Ψpl
    Ψt0 = deepcopy(Ψ)
    Ψp0 = deepcopy(Ψpl)

    sum(debug) > 0 && println("\t\tΨaxis\t\tΔΨ\t\tError")
    converged = false
    error_outer = 0.0
    update_surfaces = (compute_Ip_from === :fsa) || (profile isa Union{PressureJtoR, PressureJt})
    for j in 1:Nout
        Ψa0 = canvas.Ψaxis
        invert_GS_zero_bnd!(canvas); # this defines U for the boundary integral
        set_boundary_flux!(canvas, j==1 ? 1.0 : relax)
        for i in 1:Nin
            Ψai = canvas.Ψaxis
            Ψt0 .= Ψ
            Ψp0 .= Ψpl
            invert_GS!(canvas; update_Ψitp=false)  # interpolation updated after relaxation
            if !initialize_current || (i != 1)
                @. Ψ   = (1.0 - relax) * Ψt0 + relax * Ψ
                @. Ψpl = (1.0 - relax) * Ψp0 + relax * Ψpl
            end
            update_bounds!(canvas; update_Ψitp=true)
            Jtor!(canvas, profile; update_surfaces=(j==1 && i==1), compute_Ip_from)
            error_inner = abs((canvas.Ψaxis - Ψai) / (relax * Ψai))
            if sum(debug) == 2
                println("\tInner $(i):\t", canvas.Ψaxis, "\t", canvas.Ψbnd - canvas.Ψaxis, "\t", error_inner)
            end
        end
        if control === :shape
            shape_control!(canvas, fixed_coils, Acps, b_offset)
        elseif control === :radial
            radial_feedback!(canvas, Rtarget, 0.5)
        elseif control === :vertical
            vertical_feedback!(canvas, Ztarget, 0.5)
        elseif control === :position
            axis_feedback!(canvas, Rtarget, Ztarget, 0.5)
        elseif control === :eddy
            eddy_control!(canvas)
        elseif control === :implicit
            implicit_eddy!(canvas, profile)
        end

        sync_Ψ!(canvas; update_Ψitp=true)
        update_bounds!(canvas; update_Ψitp=false)
        Jtor!(canvas, profile; update_surfaces, compute_Ip_from)

        error_outer = abs((canvas.Ψaxis - Ψa0) / (relax * Ψa0))
        if sum(debug) > 0
            @printf("Iteration %d:\t%.8f\t%.8f\t%e\n", j, canvas.Ψaxis, canvas.Ψbnd - canvas.Ψaxis, error_outer)
            flush(stdout)
        end
        sum(debug) == 2 && display(plot(canvas))
        if error_outer < tolerance
            converged = true
            break
        end
    end

    if !converged && tolerance > 0.0
        sum(debug) > 0 && @warn "FRESCO did not converged to $(error_outer) > $(tolerance) in $(Nout) iterations"
        return 1
    end

    return 0

end