# Initialize to uniform current in ellipse half the size of domain
function initial_current(canvas::Canvas, R::Real, Z::Real)
    Rb, Zb, Ip = canvas._Rb_target, canvas._Zb_target, canvas.Ip
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

# Based on https://arxiv.org/pdf/1503.03135
abstract type CurrentProfile end

mutable struct BetapIp{T} <: CurrentProfile
    betap::T
    alpha_m::T
    alpha_n::T
    Beta0::T
    L::T
end

mutable struct PaxisIp{T} <: CurrentProfile
    paxis::T
    alpha_m::T
    alpha_n::T
    L::T
    Beta0::T
end

const FuncInterp = Union{Function, DataInterpolations.AbstractInterpolation}

mutable struct PprimeFFprime{F1<:FuncInterp, F2<:FuncInterp} <: CurrentProfile
    pprime::F1
    ffprime::F2
    ffp_scale::Float64
end

PprimeFFprime(pprime::FuncInterp, ffprime::FuncInterp) = PprimeFFprime(pprime, ffprime, 1.0)

function PprimeFFprime(dd::IMAS.dd)
    eqt1d = dd.equilibrium.time_slice[].profiles_1d
    psi_norm = eqt1d.psi_norm
    pprime = DataInterpolations.CubicSpline(eqt1d.dpressure_dpsi, psi_norm; extrapolation=ExtrapolationType.Extension)
    ffprime = DataInterpolations.CubicSpline(eqt1d.f_df_dpsi, psi_norm; extrapolation=ExtrapolationType.Extension)
    return PprimeFFprime(pprime, ffprime)
end

mutable struct PressureJtoR{F1<:FuncInterp, F2<:FuncInterp} <: CurrentProfile
    pressure::F1
    JtoR::F2 # <Jt / R>
    J_scale::Float64
end

PressureJtoR(pressure::FuncInterp, JtoR::FuncInterp) = PressureJtoR(pressure, JtoR, 1.0)

function PressureJtoR(dd::IMAS.dd; j_p_from::Symbol=:equilibrium)
    @assert j_p_from in (:equilibrium, :core_profiles)
    eqt1d = dd.equilibrium.time_slice[].profiles_1d
    if j_p_from === :equilibrium
        psi_norm = eqt1d.psi_norm
        pressure = DataInterpolations.CubicSpline(eqt1d.pressure, psi_norm; extrapolation=ExtrapolationType.Extension)
        JtoR = DataInterpolations.CubicSpline(eqt1d.j_tor .* eqt1d.gm9, psi_norm; extrapolation=ExtrapolationType.Extension)
    else
        cp1d = dd.core_profiles.profiles_1d[]
        psi_norm = cp1d.grid.psi_norm
        gm9 = IMAS.interp1d(eqt1d.psi_norm, eqt1d.gm9).(psi_norm)
        pressure = DataInterpolations.CubicSpline(cp1d.pressure, psi_norm; extrapolation=ExtrapolationType.Extension)
        JtoR = DataInterpolations.CubicSpline(cp1d.j_tor .* gm9, psi_norm; extrapolation=ExtrapolationType.Extension)
    end
    return PressureJtoR(pressure, JtoR)
end

mutable struct PressureJt{F1<:FuncInterp, F2<:FuncInterp} <: CurrentProfile
    pressure::F1
    Jt::F2 # <Jt / R> / <1 / R>
    J_scale::Float64
end

PressureJt(pressure::FuncInterp, Jt::FuncInterp) = PressureJt(pressure, Jt, 1.0)

function PressureJt(dd::IMAS.dd; j_p_from::Symbol=:equilibrium)
    @assert j_p_from in (:equilibrium, :core_profiles)
    eqt1d = dd.equilibrium.time_slice[].profiles_1d
    if j_p_from === :equilibrium
        psi_norm = eqt1d.psi_norm
        pressure = DataInterpolations.CubicSpline(eqt1d.pressure, psi_norm; extrapolation=ExtrapolationType.Extension)
        Jt = DataInterpolations.CubicSpline(eqt1d.j_tor, psi_norm; extrapolation=ExtrapolationType.Extension)
    else
        cp1d = dd.core_profiles.profiles_1d[]
        psi_norm = cp1d.grid.psi_norm
        pressure = DataInterpolations.CubicSpline(cp1d.pressure, psi_norm; extrapolation=ExtrapolationType.Extension)
        Jt = DataInterpolations.CubicSpline(cp1d.j_tor , psi_norm; extrapolation=ExtrapolationType.Extension)
    end
    return PressureJt(pressure, Jt)
end

mutable struct SigmaQ{F1<:FuncInterp, F2<:FuncInterp} <: CurrentProfile
    sigma::F1
    q::F2
end
@inline shape_function(psi_norm::Real, profile::Union{BetapIp, PaxisIp}) = (1.0 - psi_norm ^ profile.alpha_m) ^ profile.alpha_n

function shape_integral(canvas::Canvas, profile::Union{BetapIp, PaxisIp}, psi_norm)
    am, an = profile.alpha_m, profile.alpha_n
    invam = 1.0 / am
    man = -an
    invam_p1 = 1.0 + 1.0 / am
    I = F21(invam, man, invam_p1, 1.0 ^ am) - psi_norm * F21(invam, man, invam_p1, psi_norm ^ am)
    return I * (canvas.Ψbnd - canvas.Ψaxis)
end

function BetapIp(betap,alpha_m,alpha_n)
    if alpha_m < 0 || alpha_n < 0
        @error "alpha_m/n must be positive"
    end
    return BetapIp(betap,alpha_m, alpha_n,zero(betap),zero(betap))
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

function PaxisIp(paxis, alpha_m, alpha_n)
    if alpha_m < 0 || alpha_n < 0
        @error "alpha_m/n must be positive"
    end
    return PaxisIp(paxis, alpha_m, alpha_n, zero(paxis), zero(paxis))
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

function pprime(canvas::Canvas, profile::Union{BetapIp,PaxisIp}, psi_norm)
    return (profile.L * profile.Beta0 / canvas.Raxis) * shape_function(psi_norm, profile)
end

function ffprime(canvas::Canvas, profile::Union{BetapIp,PaxisIp}, psi_norm)
    return μ₀ * profile.L * (1 - profile.Beta0) * shape_function(psi_norm, profile)
end

function pprime(canvas::Canvas, profile::PprimeFFprime, psi_norm)
    profile.pprime(psi_norm)
end

function ffprime(canvas::Canvas, profile::PprimeFFprime, psi_norm)
    profile.ffprime(psi_norm) * profile.ffp_scale
end

function pprime(canvas::Canvas, profile::Union{PressureJtoR, PressureJt}, psi_norm)
    return DataInterpolations.derivative(profile.pressure, psi_norm) / (canvas.Ψbnd - canvas.Ψaxis)
end

function ffprime(canvas::Canvas, profile::PressureJtoR, psi_norm)
    gm1 = canvas._gm1
    psin = range(0.0, 1.0, length(gm1))
    gitp = DataInterpolations.CubicSpline(gm1, psin; extrapolation=ExtrapolationType.None)
    return ffprime(canvas, profile, psi_norm, gitp(psi_norm))
end

function ffprime(canvas::Canvas, profile::PressureJtoR, psi_norm::Real, gm1::Real)
    pp = pprime(canvas, profile, psi_norm)
    return -μ₀ * (pp + profile.J_scale * profile.JtoR(psi_norm) / twopi) / gm1
end

function ffprime(canvas::Canvas, profile::PressureJt, psi_norm)
    gm1 = canvas._gm1
    gm9 = canvas._gm9
    psin = range(0.0, 1.0, length(gm1))
    gm1itp = DataInterpolations.CubicSpline(gm1, psin; extrapolation=ExtrapolationType.None)
    gm9itp = DataInterpolations.CubicSpline(gm9, psin; extrapolation=ExtrapolationType.None)
    return ffprime(canvas, profile, psi_norm, gm1itp(psi_norm), gm9itp(psi_norm))
end

function ffprime(canvas::Canvas, profile::PressureJt, psi_norm::Real, gm1::Real, gm9::Real)
    pp = pprime(canvas, profile, psi_norm)
    JtoR = profile.Jt(psi_norm) * gm9
    return -μ₀ * (pp + profile.J_scale * JtoR / twopi) / gm1
end

function Jtor!(canvas::Canvas, profile::PprimeFFprime; kwargs...)
    Rs, Zs, Ψ, Ip, Jt, is_inside = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip, canvas._Jt, canvas._is_inside
    Jt .= 0.0

    # compute the FF' contribution to Ip
    for (i,R) in enumerate(Rs)
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                Jt[i, j] = -twopi * ffprime(canvas, profile, psin) / (R * μ₀)
            end
        end
    end
    If_c = sum(Jt) * step(Rs) * step(Zs)

    # compute total Ip without scaling
    for (i,R) in enumerate(Rs)
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                Jt[i, j] += -twopi * R * pprime(canvas, profile, psin)
            end
        end
    end
    Ic = sum(Jt) * step(Rs) * step(Zs)

    # scale FF' to fix Ip
    fac =  1 + (Ip - Ic) / If_c
    for (i,R) in enumerate(Rs)
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                Jt[i, j] = -twopi * (R * pprime(canvas, profile, psin) + fac * ffprime(canvas, profile, psin) / (R * μ₀))
            end
        end
    end

    profile.ffp_scale *= fac

    return Jt
end

function Jtor!(canvas::Canvas, profile::PressureJtoR{F, F}; update_surfaces::Bool) where {F<:DataInterpolations.AbstractInterpolation}
    Rs, Zs, Ψ, Ψaxis, Ψbnd, Ip = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ψaxis, canvas.Ψbnd, canvas.Ip
    Jt, is_inside, Vp, gm1 = canvas._Jt, canvas._is_inside, canvas._Vp, canvas._gm1

    Jt .= 0.0
    FRESCO.compute_FSAs!(canvas; update_surfaces)

    x = range(0.0, 1.0, length(Vp))
    psic  = (Ψbnd - Ψaxis) .* x
    VJ = (k, xx) -> Vp[k] * profile.JtoR(x[k])
    Ic = trapz(psic, VJ) / (2π)
    profile.J_scale = Ip / Ic

    gitp =  DataInterpolations.CubicSpline(gm1, x; extrapolation=ExtrapolationType.None)

    inv_ΔΨ = 1.0 / (Ψbnd - Ψaxis)
    pprime  = x -> DataInterpolations.derivative(profile.pressure, x) * inv_ΔΨ

    for (i,R) in enumerate(Rs)
        inv_R = 1.0 / R
        R2 = R ^ 2
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                pterm = twopi * (R2 - 1.0 / gitp(psin)) * pprime(psin)
                jterm = profile.J_scale * profile.JtoR(psin) / gitp(psin)
                Jt[i, j] = -inv_R * (pterm - jterm)
            end
        end
    end

    return Jt
end

function Jtor!(canvas::Canvas, profile::PressureJt{F, F}; update_surfaces::Bool) where {F<:DataInterpolations.AbstractInterpolation}
    Rs, Zs, Ψ, Ψaxis, Ψbnd, Ip = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ψaxis, canvas.Ψbnd, canvas.Ip
    Jt, is_inside, Vp, gm1, gm9 = canvas._Jt, canvas._is_inside, canvas._Vp, canvas._gm1, canvas._gm9

    Jt .= 0.0
    FRESCO.compute_FSAs!(canvas; update_surfaces)

    x = range(0.0, 1.0, length(Vp))
    gm1itp =  DataInterpolations.CubicSpline(gm1, x; extrapolation=ExtrapolationType.None)
    gm9itp =  DataInterpolations.CubicSpline(gm9, x; extrapolation=ExtrapolationType.None)


    psic  = (Ψbnd - Ψaxis) .* x
    VJ = (k, xx) -> Vp[k] * profile.Jt(x[k]) * gm9itp(x[k])
    Ic = trapz(psic, VJ) / (2π)
    profile.J_scale = Ip / Ic

    inv_ΔΨ = 1.0 / (Ψbnd - Ψaxis)
    pprime  = x -> DataInterpolations.derivative(profile.pressure, x) * inv_ΔΨ

    for (i,R) in enumerate(Rs)
        inv_R = 1.0 / R
        R2 = R ^ 2
        for (j, z) in enumerate(Zs)
            if is_inside[i, j]
                psin = psinorm(Ψ[i, j], canvas)
                pterm = twopi * (R2 - 1.0 / gm1itp(psin)) * pprime(psin)
                jterm = profile.J_scale * profile.Jt(psin) * gm9itp(psin) / gm1itp(psin)
                Jt[i, j] = -inv_R * (pterm - jterm)
            end
        end
    end

    return Jt
end
