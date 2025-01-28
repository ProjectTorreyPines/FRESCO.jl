function solve!(canvas::Canvas, profile::CurrentProfile, Nout::Int, Nin::Int;
                Rtarget = 0.5 * sum(extrema(canvas._Rb_target)),
                Ztarget = canvas._Zb_target[argmax(canvas._Rb_target)],
                debug=0,
                relax::Real=0.5,
                tolerance::Real=0.0,
                control::Union{Nothing, Symbol}=:shape,
                fixed_coils::AbstractVector{Int}=canvas._fixed_coils,
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
    update_surfaces = profile isa Union{PressureJtoR, PressureJt}
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
            Jtor!(canvas, profile; update_surfaces=(j==1 && i==1))
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
        Jtor!(canvas, profile; update_surfaces)

        error_outer = abs((canvas.Ψaxis - Ψa0) / (relax * Ψa0))
        sum(debug) > 0 && println("Iteration $(j):\t", canvas.Ψaxis, "\t", canvas.Ψbnd - canvas.Ψaxis, "\t", error_outer)
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


# function solve_q!(canvas::Canvas, prof_sq::SigmaQ, Nout::Int, Nin::Int;
#                 Rtarget = 0.5 * sum(extrema(canvas._Rb_target)),
#                 Ztarget = canvas._Zb_target[argmax(canvas._Rb_target)],
#                 debug=0,
#                 relax::Real=0.5,
#                 tolerance::Real=0.0,
#                 control::Union{Nothing, Symbol}=:shape,
#                 initialize_current=true,
#                 initialize_mutuals=(control === :eddy))

#     @assert control in (nothing, :shape, :vertical, :radial, :position, :eddy)

#     if initialize_current
#         J = (x,y) -> initial_current(canvas, x, y)
#         gridded_Jtor!(canvas, J)
#     else
#         gridded_Jtor!(canvas)
#     end

#     set_Ψvac!(canvas)
#     Ψ, Ψpl = canvas.Ψ, canvas._Ψpl
#     Ψt0 = deepcopy(Ψ)
#     Ψp0 = deepcopy(Ψpl)

#     if initialize_mutuals
#         set_mutuals!(canvas)
#         set_flux_at_coils!(canvas)
#     end

#     pVp = plot(title="dV/dΨ", xlabel="ψₙ")
#     pgm1 = plot(title="<R⁻²>", xlabel="ψₙ")
#     pt = plot(title="dV/dΨ * <R⁻²>", xlabel="ψₙ")
#     pf = plot(title="F", xlabel="ψₙ")
#     for k in 1:3
#         Vp, gm1 = canvas._Vp, canvas._gm1
#         if k > 1
#             Vp_old, gm1_old = deepcopy(Vp), deepcopy(gm1)
#         end
#         FRESCO.compute_FSAs!(canvas; update_surfaces=true)
#         if k > 1
#             Vp .= 0.9 .* Vp_old .+ 0.1 .* Vp
#             gm1 .= 0.9 .* gm1_old .+ 0.1 .* gm1
#         end
#         psi_norm = range(0, 1, length(Vp))
#         f = 2π .* prof_sq.q.(psi_norm) ./ (Vp .* gm1)
#         plot!(pVp, psi_norm, Vp, lw=2, label="Iter $k")
#         plot!(pgm1, psi_norm, gm1, lw=2, label="Iter $k")
#         plot!(pt, psi_norm, Vp .* gm1, lw=2, label="Iter $k")
#         k>1 && plot!(pt, psi_norm, .-prof_sq.q.(psi_norm), lw=2, label="-q")
#         plot!(pf, psi_norm, f, lw=2, label="Iter $k")

#         display(plot(pVp, pgm1, pt, pf, size=(600,400), layout=(2,2)))

#         prof_pf = PprimeFFprime(canvas, prof_sq; update_FSA=false)

#         psin = range(0,1,10001)
#         R = 1.5
#         plt = plot(psin, -2π * (R * prof_pf.pprime.(psin)), lw=2, label="pprime")
#         plot!(plt, psin, -2π * (prof_pf.ffprime.(psin) / (R * μ₀)), lw=2, label="FFprime")
#         display(plt)

#         sum(debug) > 0 && println("\t\tΨaxis\t\t\tΔΨ\t\t\tError")
#         for j in 1:Nout
#             Ψa0 = canvas.Ψaxis
#             #Ψ .= 0.0
#             invert_GS_zero_bnd!(canvas); # this defines U for the boundary integral
#             set_boundary_flux!(canvas, j==1 ? 1.0 : relax)
#             for i in 1:Nin
#                 Ψai = canvas.Ψaxis
#                 Ψt0 .= Ψ
#                 Ψp0 .= Ψpl
#                 invert_GS!(canvas)
#                 if !initialize_current || (i != 1)
#                     @. Ψ   = (1.0 - relax) * Ψt0 + relax * Ψ
#                     @. Ψpl = (1.0 - relax) * Ψp0 + relax * Ψpl
#                 end
#                 update_bounds!(canvas)
#                 Jtor!(canvas, prof_pf)
#                 error_inner = abs((canvas.Ψaxis - Ψai) / (relax * Ψai))
#                 if sum(debug) == 2
#                     println("\tInner $(i):\t", canvas.Ψaxis, "\t", canvas.Ψbnd - canvas.Ψaxis, "\t", error_inner)
#                     #j>1 &&
#                     #display(plot(canvas))
#                     #display(heatmap(canvas.Rs, canvas.Zs, canvas._Jt', aspect_ratio=:equal))
#                 end
#             end
#             if (control === :shape)
#                 j == 1 && println("WARNING: Need to update definition of fixed coils as input or programmatically")
#                 fixed = vcat(1:6, 25:48)
#                 shape_control!(canvas, fixed)
#             end

#             (control === :radial) && radial_feedback!(canvas, Rtarget, 0.5)
#             (control === :vertical) && vertical_feedback!(canvas, Ztarget, 0.5)
#             (control === :position) && axis_feedback!(canvas, Rtarget, Ztarget, 0.5)
#             (control === :eddy) && eddy_control2!(canvas)
#             sync_Ψ!(canvas)
#             update_bounds!(canvas)
#             Jtor!(canvas, prof_pf)
#             error_outer = abs((canvas.Ψaxis - Ψa0) / (relax * Ψa0))
#             sum(debug) > 0 && println("Iteration $(j):\t", canvas.Ψaxis, "\t", canvas.Ψbnd - canvas.Ψaxis, "\t", error_outer)
#             sum(debug) == 2 && display(plot(canvas))
#             if error_outer < tolerance
#                 break
#             end
#         end
#         display(plot(canvas))
#     end

#     return canvas
# end

# function PprimeFFprime(canvas, prof_sq::SigmaQ; update_FSA::Bool=true)
#     update_FSA && FRESCO.compute_FSAs!(canvas; update_surfaces=true)

#     Ψaxis, Ψbnd, Vp, gm1 = canvas.Ψaxis, canvas.Ψbnd, canvas._Vp, canvas._gm1

#     psi_norm = range(0, 1, length(Vp))
#     p = prof_sq.sigma.(psi_norm) ./ (Vp .^ (5/3.))
#     f = 2π .* prof_sq.q.(psi_norm) ./ (Vp .* gm1)
#     pitp = DataInterpolations.CubicSpline(p, psi_norm; extrapolate=false)
#     fitp = DataInterpolations.CubicSpline(f, psi_norm; extrapolate=false)
#     inv_ΔΨ = 1.0 / (Ψbnd - Ψaxis)
#     pprime  = x ->           DataInterpolations.derivative(pitp, x) * inv_ΔΨ
#     ffprime = x -> fitp(x) * DataInterpolations.derivative(fitp, x) * inv_ΔΨ
#     return PprimeFFprime(pprime, ffprime)
# end

