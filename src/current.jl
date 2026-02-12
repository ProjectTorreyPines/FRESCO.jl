#******************************
# Gridded Jtor functions
#******************************

# Initialize to uniform current in ellipse half the size of domain
function initial_current(canvas::Canvas, R::Real, Z::Real)
    Rb, Zb, Ip = canvas.Rb_target, canvas.Zb_target, canvas.Ip
    Rmin, Rmax = extrema(Rb)
    Zmin, Zmax = extrema(Zb)

    R0 = 0.5 * (Rmax + Rmin)
    Z0 = 0.5 * (Zmax + Zmin)
    a = 0.4 * (Rmax - Rmin)
    b = 0.4 * (Zmax - Zmin)
    erad = ((R - R0) / a) ^ 2 + ((Z - Z0) / b) ^ 2
    if erad <= 1.0
        return Ip / (π * a * b)
    else
        return 0.0
    end
end

function gridded_Jtor!(canvas::Canvas, Jtor::Nothing)
    canvas._Jt .= 0.0
    return canvas
end

function gridded_Jtor!(canvas::Canvas, Jtor)
    Rs, Zs, Jt = canvas.Rs, canvas.Zs, canvas._Jt
    Nr, Nz = length(Rs), length(Zs)
    for (j, z) in enumerate(Zs)
        if (j === 1) || (j === Nz + 1)
            Jt[:, j] .= 0.0
        else
            for (i, r) in enumerate(Rs)
                Jt[i, j] = ((i === 1) || (i === Nr + 1)) ? 0.0 : Jtor(r, z)
            end
        end
    end
    return canvas
end

# Jt from finite difference of Ψ
function gridded_Jtor!(canvas::Canvas)
    Rs, Zs, Ψ, Jt, is_inside = canvas.Rs, canvas.Zs, canvas.Ψ, canvas._Jt, canvas._is_inside
    Jt .= 0.0
    hr, hz = step(Rs), step(Zs)
    # Compute (1/R) Δ*Ψ
    for (i, x) in enumerate(Rs)
        (i == 1 || i == length(Rs)) && continue
        for (j, y) in enumerate(Zs)
            (j == 1 || j == length(Zs)) && continue
            if is_inside[i, j]
                Jt[i, j]  = (Ψ[i, j+1] - 2 * Ψ[i, j] + Ψ[i, j-1]) / (x * hz ^ 2)
                Jt[i, j] += 2.0 * ((Ψ[i+1, j] - Ψ[i, j]) / (Rs[i+1] + x) - (Ψ[i, j] - Ψ[i-1, j]) / (x + Rs[i-1])) / (hr ^ 2)
            end
        end
    end
    Jt ./= 2π * μ₀
end

#**************************
# Current Profiles
#**************************

#*********************************************
# BetapIp and PaxisIp
# Based on https://arxiv.org/pdf/1503.03135
#*********************************************


@inline shape_function(psin::Real, profile::Union{BetapIp, PaxisIp}) = (1.0 - psin ^ profile.alpha_m) ^ profile.alpha_n

function shape_integral(canvas::Canvas, profile::Union{BetapIp, PaxisIp}, psin)
    am, an = profile.alpha_m, profile.alpha_n
    invam = 1.0 / am
    man = -an
    invam_p1 = 1.0 + 1.0 / am
    I = F21(invam, man, invam_p1, 1.0 ^ am) - psin * F21(invam, man, invam_p1, psin ^ am)
    return I * (canvas.Ψbnd - canvas.Ψaxis)
end

function BetapIp(betap,alpha_m,alpha_n)
    if alpha_m < 0 || alpha_n < 0
        @error "alpha_m/n must be positive"
    end
    return BetapIp(betap,alpha_m, alpha_n,zero(betap),zero(betap))
end

function PaxisIp(paxis, alpha_m, alpha_n)
    if alpha_m < 0 || alpha_n < 0
        @error "alpha_m/n must be positive"
    end
    return PaxisIp(paxis, alpha_m, alpha_n, zero(paxis), zero(paxis))
end

function Pprime(canvas::Canvas, profile::Union{BetapIp,PaxisIp}, psin::Real)
    return (profile.L * profile.Beta0 / canvas.Raxis) * shape_function(psin, profile)
end

function FFprime(canvas::Canvas, profile::Union{BetapIp,PaxisIp}, psin::Real)
    return μ₀ * profile.L * (1 - profile.Beta0) * shape_function(psin, profile)
end

function Jtor!(canvas::Canvas, profile::BetapIp; kwargs...)
    Rs, Zs, Ψ, Ip, Raxis, is_inside = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip, canvas.Raxis, canvas._is_inside
    ellipse = compute_ellipse(canvas)
    p_int = zero(eltype(Ψ))
    IR    = zero(eltype(Ψ))
    I_R   = zero(eltype(Ψ))
    for (i,r) in enumerate(Rs)
        r_Raxis = r / Raxis
        Raxis_r = Raxis / r
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                jtor_shape = shape_function(psin, profile)
                p_int += r * shape_integral(canvas, profile, psin)
                IR  += jtor_shape * r_Raxis
                I_R += jtor_shape * Raxis_r
            end
        end
    end
    dRdZ = step(Rs) * step(Zs)

    p_int *= dRdZ
    IR    *= dRdZ
    I_R   *= dRdZ

    LBeta0 = (-profile.betap * (μ₀ / (8π)) * Raxis * Ip ^ 2) / p_int
    L = Ip / I_R - LBeta0 * (IR / I_R - 1)
    Beta0 = LBeta0/L

    Jt = canvas._Jt
    Jt .= 0.0
    for (i,r) in enumerate(Rs)
        r_Raxis = r / Raxis
        Raxis_r = Raxis / r
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                jtor_shape = shape_function(psin, profile)
                Jt[i,j] = L * (Beta0 * r_Raxis + (1 - Beta0) * Raxis_r) * jtor_shape
            end
        end
    end
    profile.L = L
    profile.Beta0 = Beta0
    return Jt
end

function Jtor!(canvas::Canvas, profile::PaxisIp; kwargs...)
    Rs, Zs, Ψ, Ip, Raxis, is_inside = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip, canvas.Raxis, canvas._is_inside

    IR    = zero(eltype(Ψ))
    I_R   = zero(eltype(Ψ))
    for (i, r) in enumerate(Rs)
        r_Raxis = r / Raxis
        Raxis_r = Raxis / r
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                jtor_shape = shape_function(psin, profile)
                IR  += jtor_shape * r_Raxis
                I_R += jtor_shape * Raxis_r
            end
        end
    end
    dRdZ = step(Rs) * step(Zs)

    IR  *= dRdZ
    I_R *= dRdZ

    LBeta0 = -profile.paxis * Raxis / shape_integral(canvas, profile, 0.0)

    L = Ip / I_R - LBeta0 * (IR / I_R - 1)
    Beta0 = LBeta0 / L

    Jt = canvas._Jt
    Jt .= 0.0
    for (i,r) in enumerate(Rs)
        r_Raxis = r / Raxis
        Raxis_r = Raxis / r
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                jtor_shape = shape_function(psin, profile)
                Jt[i,j] = L * (Beta0 * r_Raxis + (1 - Beta0) * Raxis_r) * jtor_shape
            end
        end
    end
    profile.L = L
    profile.Beta0 = Beta0
    return Jt
end


#************************************
# Grid functions for other profiles
#************************************

check_grid(grid::Symbol) = @assert grid in (:psi_norm, :rho_tor_norm)

get_x(canvas::Canvas, profile::AbstractCurrentProfile, psin::Real) = get_x(psin, profile.grid, canvas._rho_itp)
function get_x(psin::Real, grid::Symbol, rho_itp::DataInterpolations.AbstractInterpolation)
    if grid === :psi_norm
        return psin
    elseif grid === :rho_tor_norm
        return rho_itp(psin)
    end
end

dx_dpsin(canvas::Canvas, profile::AbstractCurrentProfile, psin::Real) = dx_dpsin(psin, profile.grid, canvas._rho_itp)
function dx_dpsin(psin::Real, grid::Symbol, rho_itp::DataInterpolations.AbstractInterpolation)
    if grid === :psi_norm
        return 1.0
    elseif grid === :rho_tor_norm
        return DataInterpolations.derivative(rho_itp, psin) # drho_dpsin
    end
end


#******************
# PprimeFFprime
#******************

function Pprime(canvas::Canvas, profile::PprimeFFprime, psin::Real, x::Real=get_x(canvas, profile, psin))
    return profile.pprime(x)
end
Pprime(profile::PprimeFFprime, x::Real) = profile.pprime(x)

function FFprime(canvas::Canvas, profile::PprimeFFprime, psin::Real, x::Real=get_x(canvas, profile, psin))
    profile.ffprime(x) * profile.ffp_scale
end
FFprime(profile::PprimeFFprime, x::Real) = profile.ffprime(x) * profile.ffp_scale

function JtoR(canvas::Canvas, profile::AbstractCurrentProfile, psin::Real, x::Real=get_x(canvas, profile, psin);
              gm1=canvas._gm1_itp(psin))
    return -twopi * (Pprime(canvas, profile, psin, x) + FFprime(canvas, profile, psin, x) * gm1 / μ₀)
end

function Jtor!(canvas::Canvas, profile::PprimeFFprime; update_surfaces::Bool, compute_Ip_from::Symbol)
    Rs, Zs, Ψ, Ip, Jt, is_inside, rho_itp = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip, canvas._Jt, canvas._is_inside, canvas._rho_itp
    Ψaxis, Ψbnd = canvas.Ψaxis, canvas.Ψbnd
    Jt .= 0.0

    @assert compute_Ip_from in (:grid, :fsa)

    if compute_Ip_from === :grid
        # compute the FF' contribution to Ip
        for (i,R) in enumerate(Rs)
            for j in eachindex(Zs)
                if is_inside[i, j]
                    psin = psinorm(Ψ[i, j], Ψaxis, Ψbnd)
                    x = get_x(psin, profile.grid, rho_itp)
                    Jt[i, j] = -twopi * FFprime(profile, x) / (R * μ₀)
                end
            end
        end
        If_c = sum(Jt) * step(Rs) * step(Zs)

        # compute total Ip
        for (i,R) in enumerate(Rs)
            for j in eachindex(Zs)
                if is_inside[i, j]
                    psin = psinorm(Ψ[i, j], Ψaxis, Ψbnd)
                    x = get_x(psin, profile.grid, rho_itp)
                    Jt[i, j] += -twopi * R * Pprime(profile, x)
                end
            end
        end
        Ic = sum(Jt) * step(Rs) * step(Zs)

    elseif compute_Ip_from === :fsa
        if update_surfaces
            FRESCO.compute_FSAs!(canvas, profile; update_surfaces)
            FRESCO.update_gm9!(canvas)
            FRESCO.update_area!(canvas)
        end

        gm1, gm9, area = canvas._gm1, canvas._gm9, canvas._area

        psins = psinorm(canvas)

        # compute the FF' contribution to Ip
        Jf = (k, xx) -> -twopi * FFprime(canvas, profile, psins[k]) * gm1[k] / (μ₀ * gm9[k])
        If_c = IMAS.trapz(area, Jf)

        # compute total Ip
        Jp = (k, xx) -> -twopi * Pprime(canvas, profile, psins[k]) / gm9[k]
        Ic = If_c +  IMAS.trapz(area, Jp)
    end

    # scale FF' to fix Ip
    profile.ffp_scale *= 1 + (Ip - Ic) / If_c

    for (i,R) in enumerate(Rs)
        for j in eachindex(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], Ψaxis, Ψbnd)
                x = get_x(psin, profile.grid, rho_itp)
                Jt[i, j] = -twopi * (R * Pprime(profile, x) + FFprime(profile, x) / (R * μ₀))
            end
        end
    end

    return Jt
end


#********************************
# PressureJt and PressureJtoR
#********************************

function Pprime(canvas::Canvas, profile::Union{PressureJtoR, PressureJt}, psin::Real, x::Real=get_x(canvas, profile, psin))
    dP_dx = DataInterpolations.derivative(profile.pressure, x)
    dpsin_dΨ = 1.0 / (canvas.Ψbnd - canvas.Ψaxis)
    return dP_dx * dx_dpsin(canvas, profile, psin) * dpsin_dΨ
end
function Pprime(profile::Union{PressureJtoR, PressureJt}, x::Real, dx_dpsin::Real, Ψbnd::Real, Ψaxis::Real)
    dP_dx = DataInterpolations.derivative(profile.pressure, x)
    return dP_dx * dx_dpsin / (Ψbnd - Ψaxis)
end

function FFprime(canvas::Canvas, profile::Union{PressureJtoR, PressureJt}, psin::Real, x::Real=get_x(canvas, profile, psin);
                 gm1=canvas._gm1_itp(psin))
    return -μ₀ * (Pprime(canvas, profile, psin, x) +
                  JtoR(canvas, profile, psin, x) / twopi) / gm1
end

function JtoR(canvas::Canvas, profile::PressureJtoR, psin::Real, x::Real=get_x(canvas, profile, psin))
    return JtoR(profile, x)
end
JtoR(profile::PressureJtoR, x::Real) = profile.J_scale * profile.JtoR(x)

function JtoR(canvas::Canvas, profile::PressureJt, psin::Real, x::Real=get_x(canvas, profile, psin);
              gm9=canvas._gm9_itp(psin))
    return JtoR(profile, x, gm9)
end
JtoR(profile::PressureJt, x::Real, gm9::Real) = profile.J_scale * profile.Jt(x) * gm9

function Jtor!(canvas::Canvas, profile::Union{PressureJtoR, PressureJt}; update_surfaces::Bool, compute_Ip_from::Symbol)
    Rs, Zs, Ψ, Ip = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip
    Jt, is_inside = canvas._Jt, canvas._is_inside

    Jt .= 0.0
    FRESCO.compute_FSAs!(canvas, profile; update_surfaces)

    @assert compute_Ip_from in (:grid, :fsa)

    if compute_Ip_from === :grid
        # compute total current
        for (i,R) in enumerate(Rs)
            inv_R = 1.0 / R
            R2 = R ^ 2
            for j in eachindex(Zs)
                if is_inside[i, j]
                    psin = psinorm(Ψ[i, j], canvas)
                    gm1_psin = canvas._gm1_itp(psin)
                    x = get_x(canvas, profile, psin)
                    pterm = twopi * (R2 - 1.0 / gm1_psin) * Pprime(canvas, profile, psin, x)
                    jterm = JtoR(canvas, profile, psin, x) / gm1_psin
                    Jt[i, j] = -inv_R * (pterm - jterm)
                end
            end
        end
        Ic = sum(Jt) * step(Rs) * step(Zs)

    elseif compute_Ip_from === :fsa
        if update_surfaces
            (profile isa PressureJtoR) && FRESCO.update_gm9!(canvas)
            FRESCO.update_area!(canvas)
        end

        gm9, area = canvas._gm9, canvas._area

        # compute total Ip
        psins = psinorm(canvas)
        J = (k, xx) -> JtoR(canvas, profile, psins[k]) / gm9[k]
        Ic = IMAS.trapz(area, J)
    end

    profile.J_scale *= Ip / Ic

    for (i,R) in enumerate(Rs)
        inv_R = 1.0 / R
        R2 = R ^ 2
        for j in eachindex(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                gm1_psin = canvas._gm1_itp(psin)
                x = get_x(canvas, profile, psin)
                pterm = twopi * (R2 - 1.0 / gm1_psin) * Pprime(canvas, profile, psin, x)
                jterm = JtoR(canvas, profile, psin, x) / gm1_psin
                Jt[i, j] = -inv_R * (pterm - jterm)
            end
        end
    end

    return Jt
end