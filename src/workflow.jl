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

    @assert control in (nothing, :shape, :vertical, :radial, :position, :eddy, :magnetics)

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

    if control in [:shape, :magnetics]
        coils, flux_cps = canvas.coils, canvas.flux_cps
        ctrl_kwargs = (control === :shape) ?
            (; flux_cps, iso_cps = canvas.iso_cps, saddle_cps = canvas.saddle_cps) :
            (; flux_cps, iso_cps = canvas.loop_cps, field_cps = canvas.field_cps)
        @views active_coils = isempty(fixed_coils) ? coils : coils[setdiff(eachindex(coils), fixed_coils)]
        Acps = VacuumFields.define_A(active_coils; ctrl_kwargs...)
        b_offset = zeros(size(Acps, 1))
        fcs = @views coils[fixed_coils]
        VacuumFields.offset_b!(b_offset; fixed_coils=fcs, ctrl_kwargs...)

        return _solve!(canvas, profile, Nout, Nin; debug, relax, tolerance, control, compute_Ip_from, initialize_current, fixed_coils, Acps, b_offset)

    elseif control in (:vertical, :radial, :position)
        return _solve!(canvas, profile, Nout, Nin; debug, relax, tolerance, control, compute_Ip_from, initialize_current, Rtarget, Ztarget)

    else
        return _solve!(canvas, profile, Nout, Nin; debug, relax, tolerance, control, compute_Ip_from, initialize_current)
    end
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
        elseif control === :magnetics
            magnetics!(canvas, fixed_coils, Acps, b_offset)
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
        sum(debug) > 0 && @warn "FRESCO did not converge to $(error_outer) > $(tolerance) in $(Nout) iterations"
        return 1
    end

    return 0

end
