function Pinata(mesh::Grid, deg=2)

    ip_geo = Lagrange{2, RefTetrahedron, 1}()
    ip_fe = Lagrange{2, RefTetrahedron, deg}()
    qr = QuadratureRule{2, RefTetrahedron}(2)
    qr_face = QuadratureRule{1, RefTetrahedron}(2)
    cellvalues = CellScalarValues(qr, ip_fe, ip_geo);
    facevalues = FaceScalarValues(qr_face, ip_fe, ip_geo);

    dh = DofHandler(mesh)
    add!(dh, :Ψ, 1, ip_fe)
    close!(dh)

    u = zeros(ndofs(dh))
    K = create_symmetric_sparsity_pattern(dh)
    f = zeros(ndofs(dh))
    global_dofs = zeros(Int, ndofs_per_cell(dh))
    n_basefuncs = getnbasefunctions(cellvalues)
    fe = zeros(n_basefuncs) # Local force vector
    Ke = zeros(n_basefuncs, n_basefuncs) # Local stiffness matrix

    ∂Ω = getfaceset(mesh, "boundary")

    return Pinata(mesh, u, dh, cellvalues, facevalues, K, f, Ke, fe, global_dofs, ∂Ω)
end

function doassemble!(pin::Pinata, rhs)

    cellvalues = pin.cellvalues
    K = pin.K
    dh = pin.dh
    f = pin.f
    global_dofs = pin.global_dofs
    fe = pin.fe # Local force vector
    Ke = pin.Ke # Local stiffness matrix

    n_basefuncs = getnbasefunctions(cellvalues)
    assembler = start_assemble(K, f)

    for cell in CellIterator(dh)
        fill!(Ke, 0)
        fill!(fe, 0)
        coords = getcoordinates(cell)
        reinit!(cellvalues, cell)
        celldofs!(global_dofs, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            coords_qp = spatial_coordinate(cellvalues, q_point, coords)
            for i in 1:n_basefuncs
                δu = shape_value(cellvalues, q_point, i)
                # ∫δu ⋅ 2π * μ₀ * Jt
                Ψ_qp = function_value(cellvalues, q_point, pin.Ψ, global_dofs)
                fe[i] += -(δu * rhs(coords_qp, )) * dΩ

                ∇δu = shape_gradient(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    ∇u = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (∇δu ⋅ ∇u / coords_qp[1]) * dΩ
                end
            end
        end

        assemble!(assembler, global_dofs, fe, Ke)
    end
    return K, f
end


function invert_GS(mesh::Grid, Ψbnd::Function, Jt::Function=x->0.0, deg=2)
    pin = Pinata(mesh, deg)
    invert_GS!(pin, Ψbnd, Jt)
    return pin
end

function invert_GS!(pin::Pinata, Ψbnd::Function, Jt::Function=x->0.0)

    dbcs = ConstraintHandler(pin.dh)
    dbc = Dirichlet(:Ψ, pin.∂Ω, x -> Ψbnd(x))
    add!(dbcs, dbc)
    close!(dbcs)

    rhs = x -> twopi * μ₀ * Jt(x)
    for i in 1:1
        doassemble!(pin, rhs);
        apply!(pin.K, pin.f, dbcs)
        pin.Ψ .= pin.K \ pin.f
    end
    return pin
end