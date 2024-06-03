function toroidal_current(pin::Pinata, cell:CellCache, q_point::Int)
    coords = getcoordinates(cell)
    reinit!(pin.cellvalues, cell)
    celldofs!(global_dofs, cell)
end