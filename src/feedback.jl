function remove_flux!(canvas::Canvas, circuit::VacuumFields.AbstractCircuit)
    Rs, Zs, Ψvac = canvas.Rs, canvas.Zs, canvas._Ψvac
    Ic = circuit.current_per_turn
    if Ic != 0.0
        for (j, z) in enumerate(Zs)
            for (i, r) in enumerate(Rs)
                Ψvac[i, j] -= VacuumFields.ψ(circuit, r, z)
            end
        end
    end
    return canvas
end

function add_flux!(canvas::Canvas, circuit::VacuumFields.AbstractCircuit)
    Rs, Zs, Ψvac = canvas.Rs, canvas.Zs, canvas._Ψvac
    Ic = circuit.current_per_turn
    if Ic != 0.0
        for (j, z) in enumerate(Zs)
            for (i, r) in enumerate(Rs)
                Ψvac[i, j] += VacuumFields.ψ(circuit, r, z)
            end
        end
    end
    return canvas
end

function vertical_feedback!(canvas::Canvas, Ztarget::Real, αstar::Real)
    Raxis, Ψitp, vs_circuit = canvas.Raxis, canvas._Ψitp, canvas._vs_circuit
    dΨ_dZ = Interpolations.gradient(Ψitp, Raxis, Ztarget)[2]
    Gz = VacuumFields.dG_dZ(vs_circuit, Raxis, Ztarget)
    dI = - αstar * dΨ_dZ / (2π * μ₀ * Gz)
    remove_flux!(canvas, vs_circuit)
    I0 = vs_circuit.current_per_turn
    Inew = I0 + dI
    VacuumFields.update_coil_currents!(vs_circuit, Inew)
    add_flux!(canvas, vs_circuit)
    sync_Ψ!(canvas)
    return canvas
end

function radial_feedback!(canvas::Canvas, Rtarget::Real, αstar::Real)
    Zaxis, Ψitp, rs_circuit = canvas.Zaxis, canvas._Ψitp, canvas._rs_circuit
    dΨ_dR = Interpolations.gradient(Ψitp, Rtarget, Zaxis)[1]
    Gr = VacuumFields.dG_dR(rs_circuit, Rtarget, Zaxis)
    dI = - αstar * dΨ_dR / (2π * μ₀ * Gr)
    remove_flux!(canvas, rs_circuit)
    I0 = rs_circuit.current_per_turn
    Inew = I0 + dI
    VacuumFields.update_coil_currents!(rs_circuit, Inew)
    add_flux!(canvas, rs_circuit)
    sync_Ψ!(canvas)
    return canvas
end

function axis_feedback!(canvas::Canvas, Rtarget::Real, Ztarget::Real, αstar::Real)
    Ψitp, vs_circuit, rs_circuit = canvas._Ψitp, canvas._vs_circuit, canvas._rs_circuit
    gradΨ = Interpolations.gradient(Ψitp, Rtarget, Ztarget)
    J = @SMatrix[VacuumFields.dG_dR(rs_circuit, Rtarget, Ztarget) VacuumFields.dG_dR(vs_circuit, Rtarget, Ztarget);
                 VacuumFields.dG_dZ(rs_circuit, Rtarget, Ztarget) VacuumFields.dG_dZ(vs_circuit, Rtarget, Ztarget)]

    dI = - αstar .* (J \ gradΨ) ./ (2π * μ₀)

    remove_flux!(canvas, rs_circuit)
    remove_flux!(canvas, vs_circuit)
    I0 = @SVector[rs_circuit.current_per_turn, vs_circuit.current_per_turn]
    Inew = I0 + dI
    VacuumFields.update_coil_currents!(rs_circuit, Inew[1])
    VacuumFields.update_coil_currents!(vs_circuit, Inew[2])
    add_flux!(canvas, rs_circuit)
    add_flux!(canvas, vs_circuit)
    sync_Ψ!(canvas)
    return canvas
end

function shape_control!(canvas::Canvas, fixed::AbstractVector{Int}, Acps::Matrix{<:Real}, b_offset::AbstractVector{<:Real})
    Rs, Zs, Ψpl, coils, iso_cps, flux_cps, saddle_cps = canvas.Rs, canvas.Zs, canvas._Ψpl, canvas.coils, canvas._iso_cps, canvas._flux_cps, canvas._saddle_cps
    Ψpl_itp = ψ_interpolant(Rs, Zs, Ψpl)
    Ψpl = (x, y) -> plasma_flux(canvas, x, y, Ψpl_itp)
    dΨpl_dR = (x, y) -> plasma_dψdR(canvas, x, y, Ψpl_itp)
    dΨpl_dZ = (x, y) -> plasma_dψdZ(canvas, x, y, Ψpl_itp)
    @views fixed_coils = coils[fixed]
    @views active_coils = isempty(fixed_coils) ? coils : coils[setdiff(eachindex(coils), fixed)]
    VacuumFields.find_coil_currents!(active_coils, Ψpl, dΨpl_dR, dΨpl_dZ; iso_cps, flux_cps, saddle_cps, fixed_coils, A=Acps, b_offset)
    set_Ψvac!(canvas)
    return canvas
end

function set_mutuals!(canvas::Canvas)
    coils, mutuals = canvas.coils, canvas._mutuals

    # store mutuals in mutuals
    for j in eachindex(coils)
        for i in eachindex(coils)
            if i < j
                mutuals[i, j] = mutuals[j, i]
            else
                mutuals[i, j] = VacuumFields.mutual(coils[i], coils[j])
            end
        end
    end
    canvas._mutuals_LU = lu(mutuals)
    return canvas
end

function eddy_control!(canvas::Canvas)
    coils, Ψ_at_coils, tmp, mutuals_LU = canvas.coils, canvas._Ψ_at_coils, canvas._tmp_Ncoils, canvas._mutuals_LU
    gridded_Jtor!(canvas)

    # tmp is flux from coils at each coil
    tmp .= Ψ_at_coils
    for k in eachindex(coils)
        tmp[k] -= plasma_flux_at_coil(k, canvas)
    end
    # poloidal flux from one coil to another is -M * current_per_turn
    # so current per turn = - M \ flux
    # here we overwrite tmp, so it winds up being current_per_turn
    ldiv!(mutuals_LU, tmp)
    tmp .*= -1
    for (k, coil) in enumerate(coils)
        VacuumFields.set_current_per_turn!(coil, tmp[k])
    end
    set_Ψvac!(canvas)
end

function fit_magnetics!(canvas::Canvas, fixed::AbstractVector{Int}, Acps::Matrix{<:Real}, b_offset::AbstractVector{<:Real})
    Rs, Zs, Ψpl, coils, loop_cps, flux_cps, field_cps = canvas.Rs, canvas.Zs, canvas._Ψpl, canvas.coils, canvas._loop_cps, canvas._flux_cps, canvas._field_cps
    Ψpl_itp = ψ_interpolant(Rs, Zs, Ψpl)
    Ψpl = (x, y) -> plasma_flux(canvas, x, y, Ψpl_itp)
    dΨpl_dR = (x, y) -> plasma_dψdR(canvas, x, y, Ψpl_itp)
    dΨpl_dZ = (x, y) -> plasma_dψdZ(canvas, x, y, Ψpl_itp)
    @views fixed_coils = coils[fixed]
    @views active_coils = isempty(fixed_coils) ? coils : coils[setdiff(eachindex(coils), fixed)]
    VacuumFields.find_coil_currents!(active_coils, Ψpl, dΨpl_dR, dΨpl_dZ; iso_cps = loop_cps, flux_cps, field_cps, fixed_coils, A=Acps, b_offset)
    set_Ψvac!(canvas)
    return canvas
end