function QED_system(dd::IMAS.dd{T}, coils::CoilVectorType) where {T<:Real}
    Qstate = QED.initialize(dd)
    η = QED.η_imas(dd)

    Mcc = [VacuumFields.mutual(c1, c2) for c1 in coils, c2 in coils]
    Ic0 = [VacuumFields.current_per_turn(c) for c in coils] #current per turn
    Rc = [VacuumFields.resistance(c) for c in coils];
    Vc0 = Ic0 .* Rc
    Vwf0 = QED.Waveform.(Vc0)

    eqt = dd.equilibrium.time_slice[]
    eqt1d = eqt.profiles_1d
    eqt2d = IMAS.findfirst(:rectangular, eqt.profiles_2d)
    Ip0 = eqt.global_quantities.ip

    # COIL MUTUALS
    # in COCOS 11, psi = -L * I
    Mpc0 = zeros(T, length(coils))
    dMpc_dt = zero(Mpc0) # How Mpc changes in time (like shape)... to test later

    # INTERNAL INDUCTANCE
    Ψ = eqt2d.psi
    Jt = eqt2d.j_tor
    Ψbnd = eqt1d.psi[end]
    Rs, Zs = eqt2d.grid.dim1, eqt2d.grid.dim2
    dR, dZ = Rs[2] - Rs[1], Zs[2] - Zs[1] # assume fixed grid
    Wp = -0.5 * dR * dZ * sum((Ψ[i, j] - Ψbnd) * Jt[i, j] for i in eachindex(Rs)[2:end], j in eachindex(Zs))
    Li0 = 2 * Wp / Ip0 ^ 2

    # EXTERNAL INDUCTANCE
    # in COCOS 11, psi = -L * I
    Le0 = -Ψbnd / Ip0
    Lp0 = (Li0 + Le0)
    dLp_dt = zero(T)

    Rp0 = zero(T)
    Vni0 = zero(T)

    Qbuild =  QED.QED_build(Ic0, Vc0, Rc, Mcc, Vni0, Rp0, Lp0, dLp_dt, Mpc0, dMpc_dt, Vwf0)

    Ip = @MVector[Ip0, Ip0]
    Ic = [Ic0, Ic0]
    Li = @MVector[Li0, Li0]
    Le = @MVector[Le0, Le0]
    Mpc = [Mpc0, Mpc0]
    Rp = @MVector[Rp0, Rp0]
    Vni = @MVector[Vni0, Vni0]
    return QED_system(; Qstate, Qbuild, η, Ip, Ic, Li, Le, Mpc, Rp, Vni)
end

function setup_Qsystem!(canvas::Canvas, profile::PressureJtoR;
                        tmax::Real,
                        Nt::Int=1,
                        V_waveforms::Vector{<:QED.Waveform}=canvas.Qsystem.Qbuild.V_waveforms,
                        preprocess_canvas=true,
                        compute_Ip_from::Symbol=:fsa)
    Qsystem = canvas.Qsystem

    if preprocess_canvas
        update_bounds!(canvas; update_Ψitp=false)
        FRESCO.compute_FSAs!(canvas, profile; update_surfaces=true)
        Jtor!(canvas, profile; update_surfaces=true, compute_Ip_from)
    end

    Qsystem.tmax = tmax
    Qsystem.Nt = Nt

    update_Qsystem!(canvas, profile) # this sets all the values in [2]

    @. Qsystem.Ic[1] = Qsystem.Ic[2] = VacuumFields.current_per_turn(canvas.coils)

    # now copy values in [2] to [1] for initialization
    Qsystem.Ip[1]   = Qsystem.Ip[2]
    Qsystem.Mpc[1] .= Qsystem.Mpc[2]
    Qsystem.Li[1]   = Qsystem.Li[2]
    Qsystem.Le[1]   = Qsystem.Le[2]
    Qsystem.Rp[1]   = Qsystem.Rp[2]
    Qsystem.Vni[1]  = Qsystem.Vni[2]

    update_Qbuild!(Qsystem; V_waveforms)

    return
end

function update_Qsystem!(canvas::Canvas, profile::PressureJtoR)
    @assert !isnothing(canvas.Qsystem)

    Rs, Zs, Ψ, Ψbnd, Jt, Ip, coils = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ψbnd, canvas._Jt, canvas.Ip, canvas.coils
    is_inside, gm1_itp, gm2p_itp, Fpol_itp, rho_itp = canvas._is_inside, canvas._gm1_itp, canvas._gm2p_itp, canvas._Fpol_itp, canvas._rho_itp
    Qsystem = canvas.Qsystem

    Qsystem.Ip[2] = QED.Ip(Qsystem.Qstate)
    for k in eachindex(coils)
        Qsystem.Mpc[2][k] = -FRESCO.plasma_flux_at_coil(k, canvas) / Ip
    end

    dR, dZ = Base.step(Rs), Base.step(Zs)
    Wp = -0.5 * dR * dZ * sum((Ψ[i, j] - Ψbnd) * Jt[i, j] for i in eachindex(Rs), j in eachindex(Zs))
    Qsystem.Li[2] = 2 * Wp / Ip ^ 2
    Qsystem.Le[2] = -(Ψbnd + sum(Qsystem.Mpc[1][k] * Qsystem.Ic[1][k] for k in eachindex(coils))) / Ip

    Rp = 0.0
    Vni = 0.0
    η = Qsystem.η
    JBni_itp = Qsystem.Qstate.JBni
    for (i,R) in enumerate(Rs)
        for j in eachindex(Zs)
            if is_inside[i, j]
                psin = FRESCO.psinorm(Ψ[i, j], canvas)
                rho = rho_itp(psin)
                eta = η(rho)

                Rp += eta * Jt[i, j] ^ 2

                F = Fpol_itp(psin)
                B2 = gm2p_itp(psin) / (2π) ^ 2 + F ^ 2 * gm1_itp(psin)
                Jni =  (R ^ 2 - F^2 / B2) * FRESCO.Pprime(canvas, profile, psin)
                if !isnothing(JBni_itp)
                    Jni -= F * JBni_itp(rho) / (2π * B2)
                end
                Jni *= -2π / R
                Vni += eta * Jni * Jt[i, j]

            end
        end
    end
    Qsystem.Rp[2]  = dR * dZ * Rp / Ip ^ 2
    Qsystem.Vni[2] = dR * dZ * Vni / Ip

    return
end

function update_Qbuild!(Qsystem::QED_system; V_waveforms = Qsystem.Qbuild.V_waveforms)
    Qbuild, Ic, Li, Le, Mpc, Rp, Vni = Qsystem.Qbuild, Qsystem.Ic, Qsystem.Li, Qsystem.Le, Qsystem.Mpc, Qsystem.Rp, Qsystem.Vni
    @assert length(V_waveforms) == length(Qbuild.V_waveforms)

    Qbuild.Ic .= Ic[1] # this is the initial current for the coils
    # Qbuild.Rc and Qbuild.Mcc don't change

    # these all are taken at the halfway point
    Qbuild.Vni = 0.5 * sum(Vni)
    Qbuild.Rp = 0.5 * sum(Rp)
    Lp = Li + Le
    Qbuild.Lp = 0.5 * sum(Lp)
    @. Qbuild.Mpc = 0.5 * (Mpc[1] + Mpc[2])

    # finite difference for dLp_dt and dMpc_dt
    Qbuild.dLp_dt = (Lp[2] - Lp[1]) / Qsystem.tmax
    @. Qbuild.dMpc_dt = (Mpc[2] - Mpc[1]) / Qsystem.tmax

    Qbuild.V_waveforms .= V_waveforms
    QED.update_voltages!(Qbuild, 0.0)
    return
end