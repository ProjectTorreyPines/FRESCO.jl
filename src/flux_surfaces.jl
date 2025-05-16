function find_axis(canvas::Canvas; update_Ψitp::Bool=true)
    update_Ψitp && update_interpolation!(canvas)
    Rs, Zs, Ψ, Ip, Ψitp, is_in_wall = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Ip, canvas._Ψitp, canvas._is_in_wall
    psisign = sign(Ip)

    # find initial guess
    ia, ja = 0, 0
    found = false
    for j in eachindex(Zs)[2:end-1]
        for i in eachindex(Rs)[2:end-1]
            !is_in_wall[i, j] && continue
            P = psisign * Ψ[i, j]
            if P == minimum(p -> psisign * p, @view(Ψ[i-1:i+1, j-1:j+1]))
                if !found
                    ia, ja = i, j
                    found = true
                else
                    display(plot(canvas))
                    throw("Found multiple minimum Ψ points: $((ia, ja)) and $((i, j))")
                end
            end
        end
    end
    Rg, Zg = Rs[ia], Zs[ja]
    Raxis, Zaxis = IMAS.find_magnetic_axis(Rs, Zs, Ψitp, psisign; rguess=Rg, zguess=Zg)
    return Raxis, Zaxis, Ψitp(Raxis, Zaxis)
end

function flux_bounds!(canvas::Canvas; update_Ψitp::Bool=true)
    update_Ψitp && update_interpolation!(canvas)
    Rs, Zs, Ψ, Rw, Zw, Ψitp, r_cache, z_cache = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Rw, canvas.Zw, canvas._Ψitp, canvas._r_cache, canvas._z_cache
    Raxis, Zaxis, Ψaxis = find_axis(canvas; update_Ψitp=false)
    # BCL 7/31/24 - Potential error here with Ψbnd as that's supposed to be the "original" psi on the boundary
    #               I'm not sure how to handle that, though hopefully it gets fixed on the IMAS side
    axis2bnd = (canvas.Ip >= 0.0) ? :increasing : :decreasing
    Ψbnd = IMAS.find_psi_boundary(Rs, Zs, Ψ, Ψaxis, axis2bnd, Raxis, Zaxis, Rw, Zw, r_cache, z_cache;
                                  PSI_interpolant=Ψitp, raise_error_on_not_open=false, raise_error_on_not_closed=false).last_closed
    return canvas.Raxis, canvas.Zaxis, canvas.Ψaxis, canvas.Ψbnd = Raxis, Zaxis, Ψaxis, Ψbnd
end

psinorm(psi::Real, canvas::Canvas) = (psi - canvas.Ψaxis) / (canvas.Ψbnd - canvas.Ψaxis)
psinorm(canvas::Canvas) = range(0.0, 1.0, length(canvas._surfaces))

function boundary!(canvas::Canvas)
    Rs, Zs, Ψ, Raxis, Zaxis, Ψaxis, Ψbnd = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Raxis, canvas.Zaxis, canvas.Ψaxis, canvas.Ψbnd
    is_inside, r_cache, z_cache = canvas._is_inside, canvas._r_cache, canvas._z_cache
    #B = IMAS.flux_surface(Rs, Zs, Ψ, Raxis, Zaxis, Float64[], Float64[], Ψbnd, :closed)
    #@assert length(B) == 1
    #r, z = B[1].r, B[1].z
    r, z = IMASutils.contour_from_midplane!(r_cache, z_cache, Ψ, Rs, Zs, Ψbnd, Raxis, Zaxis, Ψaxis)
    canvas._bnd = [@SVector[r[k], z[k]] for k in eachindex(r)]
    canvas._rextrema = extrema(r)
    canvas._zextrema = extrema(z)

    ellipse = compute_ellipse(canvas)
    for (j, z) in enumerate(Zs)
        for (i, r) in enumerate(Rs)
            psin = psinorm(Ψ[i, j], canvas)
            is_inside[i, j] = in_core(r, z, psin, canvas, ellipse)
        end
    end
end

function update_bounds!(canvas; update_Ψitp::Bool=true)
    flux_bounds!(canvas; update_Ψitp)
    boundary!(canvas)
    return canvas
end

function in_plasma_bb(r::Real, z::Real, canvas::Canvas)
    rmin, rmax = canvas._rextrema
    (r < rmin || r > rmax) && return false

    zmin, zmax = canvas._zextrema
    (z < zmin || z > zmax) && return false

    return true
end

function in_core(r::Real, z::Real, psin::Real, canvas::Canvas,
    ellipse::Union{Nothing,AbstractVector{<:Real}}=nothing)

    # Check psinorm value
    psin > 1.0 && return false

    # Check outside bounding box
    !in_plasma_bb(r, z, canvas) && return false

    in_ellipse(r, z, ellipse) && return true

    # Finally make sure it's in the boundary
    return inpolygon((r, z), canvas._bnd) == 1
end

in_ellipse(r, z, ellipse::Nothing) = false

function in_ellipse(r::Real, z::Real, ellipse::AbstractVector{<:Real})
    @assert length(ellipse) == 4
    R0, Z0, a, b = ellipse
    return ((r - R0) / a)^2 + ((z - Z0) / b)^2 <= 1.0
end

function compute_ellipse(canvas::Canvas)
    bnd, rext, zext = canvas._bnd, canvas._rextrema, canvas._zextrema
    κ = (zext[2] - zext[1]) / (rext[2] - rext[1])
    R0, Z0 = centroid(bnd)
    radius = p -> sqrt((p[1] - R0)^2 + ((p[2] - Z0) / κ)^2)
    a = minimum(radius, bnd)
    b = κ * a
    return @SVector[R0, Z0, a, b]
end


function trace_surfaces!(canvas::Canvas)
    Rs, Zs, Ψ, Raxis, Zaxis, Ψaxis = canvas.Rs, canvas.Zs, canvas.Ψ, canvas.Raxis, canvas.Zaxis, canvas.Ψaxis
    Ψbnd, Ψitp, Rw, Zw, surfaces = canvas.Ψbnd, canvas._Ψitp, canvas.Rw, canvas.Zw, canvas._surfaces
    r_cache, z_cache = canvas._r_cache, canvas._z_cache
    psi_levels = range(Ψaxis, Ψbnd, length(surfaces))
    return IMAS.trace_simple_surfaces!(surfaces, psi_levels, Rs, Zs, Ψ, Ψitp, Raxis, Zaxis, Rw, Zw, r_cache, z_cache)
end

function update_Vp!(canvas::Canvas)
    Ip, Vp, surfaces = canvas.Ip, canvas._Vp, canvas._surfaces
    sign_dpsi = sign(Ip)
    Threads.@threads for k in eachindex(surfaces)
        Vp[k] = sign_dpsi * surfaces[k].int_fluxexpansion_dl
    end
    x = psinorm(canvas)
    canvas._Vp_itp = DataInterpolations.CubicSpline(Vp, x; extrapolation=ExtrapolationType.None)
    return canvas
end

gm1_integrand(j, surface) = surface.fluxexpansion[j] / surface.r[j] ^ 2
function update_gm1!(canvas::Canvas)
    gm1, surfaces = canvas._gm1, canvas._surfaces
    Threads.@threads for k in eachindex(surfaces)
        f1 = (j, xx) -> gm1_integrand(j, surfaces[k])
        gm1[k] = IMAS.flux_surface_avg(f1, surfaces[k])
    end
    x = psinorm(canvas)
    canvas._gm1_itp = DataInterpolations.CubicSpline(gm1, x; extrapolation=ExtrapolationType.None)
    return canvas
end

gm9_integrand(j, surface) = surface.fluxexpansion[j] / surface.r[j]
function update_gm9!(canvas::Canvas)
    gm9, surfaces = canvas._gm9, canvas._surfaces
    Threads.@threads for k in eachindex(surfaces)
        f9 = (j, xx) -> gm9_integrand(j, surfaces[k])
        gm9[k] = IMAS.flux_surface_avg(f9, surfaces[k])
    end
    x = psinorm(canvas)
    canvas._gm9_itp = DataInterpolations.CubicSpline(gm9, x; extrapolation=ExtrapolationType.None)
    return canvas
end

function update_Fpol!(canvas::Canvas, profile::AbstractCurrentProfile)
    Ψaxis, Ψbnd, Fbnd, Fpol = canvas.Ψaxis, canvas.Ψbnd, canvas.Fbnd, canvas._Fpol
    x = psinorm(canvas)
    psi1d = range(Ψaxis, Ψbnd, length(x))
    # starts as F^2
    IMAS.cumtrapz!(Fpol, psi1d, FFprime.(Ref(canvas), Ref(profile), x))
    Fpol .= 2 .* Fpol .- Fpol[end] .+ Fbnd^2
    Fpol .= sign(Fbnd) .* sqrt.(Fpol) # now take sqrt with proper sign
    canvas._Fpol_itp = DataInterpolations.CubicSpline(Fpol, x; extrapolation=ExtrapolationType.None)
    return canvas
end

function update_rho!(canvas::Canvas)

    rho, Ψaxis, Ψbnd, Vp, gm1, Fpol = canvas._rho, canvas.Ψaxis, canvas.Ψbnd, canvas._Vp, canvas._gm1, canvas._Fpol
    x = psinorm(canvas)
    psi1d = range(Ψaxis, Ψbnd, length(x))

    # compute toroidal flux
    #   Φ = ∫dΨ F * dV/dΨ * <R⁻²> / 2π
    # we ignore the 2π because we'll normalize anyway
    IMAS.cumtrapz!(rho, psi1d, Fpol .* Vp .* gm1)

    # then turn into rho_tor_norm
    rho .= sqrt.(rho ./ rho[end])
    rho[1] = 0.0
    rho[end] = 1.0

    canvas._rho_itp = DataInterpolations.CubicSpline(rho, x; extrapolation=ExtrapolationType.None)
end

# This is <|∇Ψ|^2 / R^2>, so like the gm2 metric in IMAS but for Ψ not rho_tor
function update_gm2p!(canvas::Canvas, profile::AbstractCurrentProfile)
    gm2p, Ψaxis, Ψbnd, Vp = canvas._gm2p, canvas.Ψaxis, canvas.Ψbnd, canvas._Vp
    x = psinorm(canvas)
    fac = twopi * μ₀ * (Ψbnd - Ψaxis)
    IMAS.cumtrapz!(gm2p, x, Vp .* JtoR.(Ref(canvas), Ref(profile), x))
    gm2p .*= fac ./ Vp
    canvas._gm2p_itp = DataInterpolations.CubicSpline(gm2p, x; extrapolation=ExtrapolationType.None)
    return canvas
end

function update_area!(canvas::Canvas)
    area, Ψaxis, Ψbnd, Vp, gm9 = canvas._area, canvas.Ψaxis, canvas.Ψbnd, canvas._Vp, canvas._gm9
    x = psinorm(canvas)
    psi1d = range(Ψaxis, Ψbnd, length(x))
    darea_dpsi = (k, x) -> Vp[k] * gm9[k] / 2π
    IMAS.cumtrapz!(area, psi1d, darea_dpsi)
    canvas._area_itp = DataInterpolations.CubicSpline(area, x; extrapolation=ExtrapolationType.None)
    return canvas
end

function _update_common_fsas!(canvas::Canvas, profile::AbstractCurrentProfile; update_surfaces=false, control::Symbol=:eddy)
    update_surfaces && trace_surfaces!(canvas)
    update_Vp!(canvas)
    update_gm1!(canvas)
    update_Fpol!(canvas, profile)
    update_rho!(canvas)
    (control === :implicit_eddy) && update_gm2p!(canvas, profile)
    return canvas
end

compute_FSAs!(canvas::Canvas, profile::AbstractCurrentProfile; update_surfaces=false, control::Symbol=:eddy) =  _update_common_fsas!(canvas, profile; update_surfaces, control)

function compute_FSAs!(canvas::Canvas, profile::PressureJt; update_surfaces=false, control::Symbol=:eddy)
    _update_common_fsas!(canvas, profile; update_surfaces, control)
    update_gm9!(canvas)
    return canvas
end