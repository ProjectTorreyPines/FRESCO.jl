function solve!(C::Canvas, profile::CurrentProfile, out::Int, in::Int; debug=false)
    J = (x,y) -> initial_current(C, x, y)
    gridded_Jtor!(C, J)
    set_Ψvac!(C)
    Ψ, Ψpl = C.Ψ, C._Ψpl
    Ψt0 = deepcopy(Ψ)
    Ψp0 = deepcopy(Ψpl)
    println("\t\tΨaxis\t\tError")
    for j in 1:out
        Ψa0 = C.Ψaxis
        #Ψ .= 0.0
        invert_GS_zero_bnd!(C); # this defines U for the boundary integral
        set_boundary_flux!(C)
        for i in 1:in
            Ψt0 .= Ψ
            Ψp0 .= Ψpl
            invert_GS!(C)
            if (i != 1.0)
                relax = 0.1
                @. Ψ   = (1.0 - relax) * Ψt0 + relax * Ψ
                @. Ψpl = (1.0 - relax) * Ψp0 + relax * Ψpl
            end
            update_bounds!(C)
            Jtor!(C, profile)
        end
        #vertical_feedback!(C, axis.z, 9, 18, 1e5)
        #radial_feedback!(C, axis.r, @SVector[6, 15], 1e5)
        #print(C.Ψbnd, "\t")
        shape_control!(C)
        sync_Ψ!(C)
        update_bounds!(C)
        Jtor!(C, profile)
        #println(C.Ψbnd)
        if debug
            @show C.Raxis, C.Zaxis, C.Ψaxis
            display(plot(C))
        end
        println("Iteration $(j):\t", C.Ψaxis, "\t", abs((C.Ψaxis - Ψa0) / Ψa0))
    end
    return C
end