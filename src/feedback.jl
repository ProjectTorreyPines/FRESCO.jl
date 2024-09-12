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

function shape_control!(canvas::Canvas, fixed::AbstractVector{Int}=Int[])
    Rs, Zs, Ψpl, coils, Rb, Zb = canvas.Rs, canvas.Zs, canvas._Ψpl, canvas.coils, canvas._Rb_target, canvas._Zb_target
    Ψpl_itp = ψ_interpolant(Rs, Zs, Ψpl)
    iso_cps = VacuumFields.IsoControlPoints(Rb, Zb)
    @views fixed_coils = coils[fixed]
    @views active_coils = isempty(fixed_coils) ? coils : coils[setdiff(eachindex(coils), fixed)]
    VacuumFields.find_coil_currents!(active_coils, Ψpl_itp; iso_cps, fixed_coils, λ_regularize=1e-12)
    set_Ψvac!(canvas)
    return canvas
end