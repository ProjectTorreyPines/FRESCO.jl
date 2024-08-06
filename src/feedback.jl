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


    I0 = VacuumFields.current(coil_p)
    #@show I0, dI
    VacuumFields.set_current!(coil_p, I0 + dI)
    #dIdZ = dΨdZ / Green(coil_m, Raxis, Ztarget)
    #α = - αstar / (Green(coil_m, R1, Z1)) - Green(coil_m, R2, Z2)
    #dI = α * (Ψ1 - Ψ2)
    #dI = α * dIdZ * dZ
    I0 = VacuumFields.current(coil_m)
    #@show I0, dI
    VacuumFields.set_current!(coil_m, I0 - dI)

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
        I0 = VacuumFields.current(coils[ic])
        #@show I0, dI
        VacuumFields.set_current!(coils[ic], I0 + dI)
    end

end