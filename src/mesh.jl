function create_mesh(dd::IMAS.dd, lc=1.0)
    ipl = IMAS.get_build_index(dd.build.layer; type=IMAS._plasma_)
    return create_mesh(dd.build.layer[ipl].outline.r, dd.build.layer[ipl].outline.z, lc)
end

function create_mesh(r, z, lc=1.0)

    gmsh.initialize(["-v","0"])
    gmsh.option.setNumber("General.Terminal",0)
    name = tempname()
    gmsh.model.add(name)
    factory = gmsh.model.geo

    p_t = 0  # point tag
    l_t = 0  # line/spline/curve tag
    c_t = 0  # curve loop tag
    s_t = 0  # surface tag

    p_t_prev = 0
    l_t_prev = 0

    N = length(r)
    (r[1] ≈ r[end]) && (z[1] ≈ z[end]) && (N -= 1)  # ignore endpoint if the same as first

    for i = 1:N
        p_t += 1
        factory.addPoint(r[i], z[i], 0.0, lc, p_t)
    end

    for i=1:N
        l_t += 1
        factory.addLine(p_t_prev + i, p_t_prev + mod(i,N) + 1, l_t)
    end
    c_t += 1
    factory.addCurveLoop(collect((l_t_prev+1):l_t),c_t)

    s_t += 1

    factory.addPlaneSurface(collect(c_t:-1:1),s_t)

    factory.synchronize()

    gmsh.model.addPhysicalGroup(1, collect((l_t_prev+1):l_t), 1, "boundary")

    gmsh.model.mesh.generate(2)

    gmsh.model.mesh.reverse()
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()

    dim = Int64(gmsh.model.getDimension())
    facedim = dim - 1

    # transfer the gmsh information
    nodes = tonodes()
    elements, gmsh_elementidx = toelements(dim)
    cellsets = tocellsets(dim, gmsh_elementidx)

    boundarydict = toboundary(facedim)
    facesets = tofacesets(boundarydict, elements)
    gmsh.finalize()

    return Grid(elements, nodes, facesets=facesets, cellsets=cellsets)
end