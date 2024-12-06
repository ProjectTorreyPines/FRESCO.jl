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

function compute_FSAs!(canvas::Canvas; update_surfaces=false)
    update_surfaces && trace_surfaces!(canvas)
    surfaces, Vp, gm1, gm9 = canvas._surfaces, canvas._Vp, canvas._gm1, canvas._gm9

    sign_dpsi = sign(surfaces[end].psi - surfaces[1].psi)

    for (k, surface) in enumerate(surfaces)

        # Vp = dvolume_dpsi
        Vp[k] = sign_dpsi * surface.int_fluxexpansion_dl

        # gm1 = <1/R^2>
        f1 = (j, xx) ->  surface.fluxexpansion[j] / surface.r[j] ^ 2
        gm1[k] = IMAS.flux_surface_avg(f1, surface)

        # gm9 = <1/R>
        f9 = (j, xx) ->  surface.fluxexpansion[j] / surface.r[j]
        gm9[k] = IMAS.flux_surface_avg(f9, surface)
    end

    return Vp, gm1, gm9
end