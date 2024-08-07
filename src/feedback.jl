function remove_coil_flux!(canvas::Canvas, k::Int)
    Rs, Zs, Ψvac, coils = canvas.Rs, canvas.Zs, canvas._Ψvac, canvas.coils
    coil = coils[k]
    Ic = VacuumFields.current(coil)
    if Ic != 0.0
        for (j, z) in enumerate(Zs)
            for (i, r) in enumerate(Rs)
                Ψvac[i, j] -= VacuumFields.ψ(coil, r, z)
            end
        end
    end
    return canvas
end

function add_coil_flux!(canvas::Canvas, k::Int)
    Rs, Zs, Ψvac, coils = canvas.Rs, canvas.Zs, canvas._Ψvac, canvas.coils
    coil = coils[k]
    Ic = VacuumFields.current(coil)
    if Ic != 0.0
        for (j, z) in enumerate(Zs)
            for (i, r) in enumerate(Rs)
                Ψvac[i, j] += VacuumFields.ψ(coil, r, z)
            end
        end
    end
    return canvas
end

function vertical_feedback!(canvas::Canvas, Ztarget::Real, icp::Int, icm::Int, α::Real)
    coils, Raxis, Zaxis, Ip, Zext, Ψitp = canvas.coils, canvas.Raxis, canvas.Zaxis, canvas.Ip, canvas._zextrema, canvas._Ψitp
    coil_p = coils[icp]
    coil_m = coils[icm]
    #dZ = 0.05 * (Zext[2] - Zext[1])
    #R1, Z1 = Raxis, Ztarget + dZ
    #R2, Z2 = Raxis, Ztarget - dZ
    #Ψ1 = Ψitp(R1, Z1)
    #Ψ2 = Ψitp(R2, Z2)

    dZ = Zaxis - Ztarget
    dI = -α * dZ * sign(Ip)
    #dΨdZ = (canvas.Ψbnd - canvas.Ψaxis) / (Zext[2] - Zext[1])

    #dIdZ = dΨdZ / Green(coil_p, Raxis, Ztarget)
    #α = - αstar / (Green(coil_p, R1, Z1)) - Green(coil_p, R2, Z2)
    #dI = α * (Ψ1 - Ψ2)
    #dI = α * dIdZ * dZ

    remove_coil_flux!(canvas, icp)
    I0 = VacuumFields.current(coil_p)
    #@show I0, dI
    VacuumFields.set_current!(coil_p, I0 + dI)
    add_coil_flux!(canvas, icp)

    remove_coil_flux!(canvas, icm)
    #dIdZ = dΨdZ / Green(coil_m, Raxis, Ztarget)
    #α = - αstar / (Green(coil_m, R1, Z1)) - Green(coil_m, R2, Z2)
    #dI = α * (Ψ1 - Ψ2)
    #dI = α * dIdZ * dZ
    I0 = VacuumFields.current(coil_m)
    #@show I0, dI
    VacuumFields.set_current!(coil_m, I0 - dI)
    add_coil_flux!(canvas, icm)
    return canvas
end

function radial_feedback!(canvas::Canvas, Rtarget::Real, ics::AbstractVector{Int}, α::Real)
    coils, Raxis, Zaxis, Ip, Zext, Ψitp = canvas.coils, canvas.Raxis, canvas.Zaxis, canvas.Ip, canvas._zextrema, canvas._Ψitp
    #dZ = 0.05 * (Zext[2] - Zext[1])
    #R1, Z1 = Raxis, Ztarget + dZ
    #R2, Z2 = Raxis, Ztarget - dZ
    #Ψ1 = Ψitp(R1, Z1)
    #Ψ2 = Ψitp(R2, Z2)

    dR = Raxis - Rtarget
    #dΨdZ = (canvas.Ψbnd - canvas.Ψaxis) / (Zext[2] - Zext[1])

    #dIdZ = dΨdZ / Green(coil_p, Raxis, Ztarget)
    #α = - αstar / (Green(coil_p, R1, Z1)) - Green(coil_p, R2, Z2)
    #dI = α * (Ψ1 - Ψ2)
    #dI = α * dIdZ * dZ
    dI = -α * dR * sign(Ip)
    for ic in ics
        remove_coil_flux!(canvas, ic)
        I0 = VacuumFields.current(coils[ic])
        #@show I0, dI
        VacuumFields.set_current!(coils[ic], I0 + dI)
        add_coil_flux!(canvas, ic)
    end

end