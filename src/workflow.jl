function solve!(canvas::Canvas, profile::CurrentProfile, Nout::Int, Nin::Int;
                Rtarget = 0.5 * sum(extrema(canvas._Rb_target)),
                Ztarget = canvas._Zb_target[argmax(canvas._Rb_target)],
                debug=0,
                relax::Real=0.5,
                tolerance::Real=0.0,
                control::Union{Nothing, Symbol}=:shape,
                fixed_coils::AbstractVector{Int}=Int[],
                initialize_current=true,
                initialize_mutuals=(control === :eddy))

    @assert control in (nothing, :shape, :vertical, :radial, :position, :eddy)

    if initialize_current
        J = (x,y) -> initial_current(canvas, x, y)
        gridded_Jtor!(canvas, J)
    else
        gridded_Jtor!(canvas)
    end

    set_Ψvac!(canvas)
    Ψ, Ψpl = canvas.Ψ, canvas._Ψpl
    Ψt0 = deepcopy(Ψ)
    Ψp0 = deepcopy(Ψpl)

    if initialize_mutuals
        set_mutuals!(canvas)
        set_flux_at_coils!(canvas)
    end

    sum(debug) > 0 && println("\t\tΨaxis\t\t\tΔΨ\t\t\tError")
    converged = false
    error_outer = 0.0
    for j in 1:Nout
        Ψa0 = canvas.Ψaxis
        #Ψ .= 0.0
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
            Jtor!(canvas, profile)
            error_inner = abs((canvas.Ψaxis - Ψai) / (relax * Ψai))
            if sum(debug) == 2
                println("\tInner $(i):\t", canvas.Ψaxis, "\t", canvas.Ψbnd - canvas.Ψaxis, "\t", error_inner)
                #j>1 && display(plot(canvas))
            end
        end
        if control === :shape
            shape_control!(canvas, fixed_coils)
        elseif control === :radial
            radial_feedback!(canvas, Rtarget, 0.5)
        elseif control === :vertical
            vertical_feedback!(canvas, Ztarget, 0.5)
        elseif control === :position
            axis_feedback!(canvas, Rtarget, Ztarget, 0.5)
        elseif control === :eddy
            eddy_control!(canvas)
        end
        sync_Ψ!(canvas; update_Ψitp=true)
        update_bounds!(canvas; update_Ψitp=false)
        Jtor!(canvas, profile)
        error_outer = abs((canvas.Ψaxis - Ψa0) / (relax * Ψa0))
        sum(debug) > 0 && println("Iteration $(j):\t", canvas.Ψaxis, "\t", canvas.Ψbnd - canvas.Ψaxis, "\t", error_outer)
        sum(debug) == 2 && display(plot(canvas))
        if error_outer < tolerance
            converged = true
            break
        end
    end

    if !converged && tolerance > 0.0
        @warn "FRESCO did not converged to $(error_outer) > $(tolerance) in $(Nout) iterations"
    end

    return canvas
end