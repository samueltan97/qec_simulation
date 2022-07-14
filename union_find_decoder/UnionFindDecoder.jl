module UnionFindDecoder

include("ClusterTrees.jl")
using .ClusterTrees

function setting_up_julia_graph_and_data_structure(detector_nodes, detector_edges)
    
    g = Graph()
    node_coord_dict = Dict() 
    edge_state_table = Dict()
    
    for i in eachindex(detector_nodes)
        add_vertex!(g)
        node_coord_dict[i] = detector_nodes[1]
    end
    for i in eachindex(detector_error_edges)
        # For now, we are assuming that all edges have the same weight
        add_edge!(g, detector_error_edges[0], detector_error_edges[1])
        edge_state_table[detector_error_edges[0]][detector_error_edges[1]] = "Unoccupied"
        edge_state_table[detector_error_edges[1]][detector_error_edges[0]] = "Unoccupied"
    end

    return g, node_coord_dict, edge_state_table
end

function initialize_cluster_trees_with_detector_parts(detection_data, detector_nodes, detector_edges)
    # detection_data is an array with 0 or 1 depending on whether a syndrome was detected on the detector
    # detector_nodes is a list of tuple with the node_id as the first element and a dictionary of attributes
    # as the second element (note that they are sorted by node_id)
    
    cluster_set = Set([])
    node_dict = Dict() 
    edge_state_table = Dict()

    for i in eachindex(detection_data)
        if detection_data[i] == 1
            new_rootnode = ClusterTreeNode.ClusterTreeNode([i, node_coord_dict[i]])
            push!(cluster_set, Dict("rootnode"=>new_rootnode, "boundary_vertices"=> Set([])))
            node_dict[i] = Dict("node"=>new_rootnode, "attributes"=>detector_nodes[i][1])
    end

    for i in eachindex(detector_error_edges)
        # For now, we are assuming that all edges have the same weight
        edge_state_table[detector_error_edges[0]][detector_error_edges[1]] = "Unoccupied"
        edge_state_table[detector_error_edges[1]][detector_error_edges[0]] = "Unoccupied"
    end

    return node_dict, edge_state_table, cluster_set        
end

function decode(detector_parts)

end
        addnode!(g, )
    def detector_error_model_to_nx_graph(model: stim.DetectorErrorModel) -> nx.Graph:
    """Convert a stim error model into a NetworkX graph."""

    g = nx.Graph()
    boundary_node = model.num_detectors
    g.add_node(boundary_node, is_boundary=True, coords=[-1, -1, -1])

    def handle_error(p: float, dets: List[int], frame_changes: List[int]):
        if p == 0:
            return
        if len(dets) == 0:
            # No symptoms for this error.
            # Code probably has distance 1.
            # Accept it and keep going, though of course decoding will probably perform terribly.
            return
        if len(dets) == 1:
            dets = [dets[0], boundary_node]
        if len(dets) > 2:
            raise NotImplementedError(
                f"Error with more than 2 symptoms can't become an edge or boundary edge: {dets!r}.")
        if g.has_edge(*dets):
            edge_data = getedgedata!(g, *dets)
            old_p = edge_data["error_probability"]
            old_frame_changes = edge_data["qubit_id"]
            # If frame changes differ, the code has distance 2; just keep whichever was first.
            if issetequal(old_frame_changes, frame_changes)
                p = p * (1 - old_p) + old_p * (1 - p)
                deleteedge!(g, *dets)
        addedge!(g, dets..., Dict('weight'=> log((1 - p) / p), 'qubit_id'=>frame_changes, 'error_probability'=>p))

    function handle_detector_coords(detector, coords):
        addnode!(g, (detector, coords=coords))

    eval_model(model, handle_error, handle_detector_coords)

    return g


end