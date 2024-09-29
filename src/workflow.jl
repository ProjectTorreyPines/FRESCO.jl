function solve!(canvas::Canvas, profile::CurrentProfile, Nout::Int, Nin::Int;
                Rtarget = 0.5 * sum(extrema(canvas._Rb_target)),
                Ztarget = canvas._Zb_target[argmax(canvas._Rb_target)],
                debug=0,
                relax::Real=0.5,
                tolerance::Real=0.0,
                control::Union{Nothing, Symbol}=:shape,
                initialize=true)

    @assert control in (nothing, :shape, :vertical, :radial, :position)

    if initialize
        J = (x,y) -> initial_current(canvas, x, y)
        gridded_Jtor!(canvas, J)
    end

    set_Ψvac!(canvas)
    Ψ, Ψpl = canvas.Ψ, canvas._Ψpl
    Ψt0 = deepcopy(Ψ)
    Ψp0 = deepcopy(Ψpl)

    sum(debug) > 0 && println("\t\tΨaxis\t\t\tΔΨ\t\t\tError")
    for j in 1:Nout
        Ψa0 = canvas.Ψaxis
        #Ψ .= 0.0
        invert_GS_zero_bnd!(canvas); # this defines U for the boundary integral
        set_boundary_flux!(canvas, j==1 ? 1.0 : relax)
        for i in 1:Nin
            Ψai = canvas.Ψaxis
            Ψt0 .= Ψ
            Ψp0 .= Ψpl
            invert_GS!(canvas)
            if (i != 1.0)
                @. Ψ   = (1.0 - relax) * Ψt0 + relax * Ψ
                @. Ψpl = (1.0 - relax) * Ψp0 + relax * Ψpl
            end
            update_bounds!(canvas)
            Jtor!(canvas, profile)
            error_inner = abs((canvas.Ψaxis - Ψai) / (relax * Ψai))
            if sum(debug) == 2
                println("\tInner $(i):\t", canvas.Ψaxis, "\t", canvas.Ψbnd - canvas.Ψaxis, "\t", error_inner)
                #j>1 && display(plot(canvas))
            end
        end
        if (control === :shape)
            j == 1 && println("WARNING: Need to update definition of fixed coils as input or programmatically")
            fixed = 1:6
            shape_control!(canvas, fixed)
        end

        (control === :radial) && radial_feedback!(canvas, Rtarget, 0.5)
        (control === :vertical) && vertical_feedback!(canvas, Ztarget, 0.5)
        (control === :position) && axis_feedback!(canvas, Rtarget, Ztarget, 0.5)

        sync_Ψ!(canvas)
        update_bounds!(canvas)
        Jtor!(canvas, profile)
        error_outer = abs((canvas.Ψaxis - Ψa0) / (relax * Ψa0))
        sum(debug) > 0 && println("Iteration $(j):\t", canvas.Ψaxis, "\t", canvas.Ψbnd - canvas.Ψaxis, "\t", error_outer)
        sum(debug) == 2 && display(plot(canvas))
        if error_outer < tolerance
            break
        end
    end

    return canvas
end