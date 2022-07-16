module UnionFindDecoder

include("ClusterTrees.jl")
using .ClusterTrees

function initialize_cluster_trees_with_detector_parts(detection_data, detector_nodes, detector_edges)
    # detection_data is an array with 0 or 1 depending on whether a syndrome was detected on the detector
    # detector_nodes is a list of tuple with the node_id as the first element and a dictionary of attributes
    # as the second element (note that they are sorted by node_id)
    
    cluster_set = Set([])
    node_dict = Dict() 
    edge_state_table = Dict()

    for i in eachindex(detection_data)
        if detection_data[i] == 1
            new_rootnode = ClusterTreeNode.ClusterTreeNode(i)
            push!(cluster_set, Dict("rootnode"=>new_rootnode, "boundary_vertices"=> Set([new_rootnode])))
            node_dict[i] = Dict("node"=>new_rootnode, "attributes"=>detector_nodes[i][1])
        else
            new_node = ClusterTreeNode.ClusterTreeNode(i)
            node_dict[i] = Dict("node"=>new_node, "attributes"=>detector_nodes[i][1])
        end
    end

    for i in eachindex(detector_error_edges)
        # For now, we are assuming that all edges have the same weight
        vertex1 = get!(edge_state_table, convert(Int64, detector_error_edges[0]), Dict())
        vertex1[convert(Int64, detector_error_edges[1])] = "Unoccupied"
        vertex2 = get!(edge_state_table, convert(Int64, detector_error_edges[1]), Dict())
        vertex2[convert(Int64, detector_error_edges[0])] = "Unoccupied"
    end

    return node_dict, edge_state_table, cluster_set        
end

function decode(detection_data, detector_nodes, detector_edges)
    node_dict, edge_state_table, cluster_set = initialize_cluster_trees_with_detector_parts(detection_data, detector_nodes, detector_edges)
    while !isempty(cluster_set)
        fusion_list = []
        for i in eachindex(cluster_set)
            boundary_vertices = cluster_set[i]["boundary_vertices"]
            for j in eachindex(boundary_vertices)
                boundary_vertex = boundary_vertices[j]
                for k in eachindex(edge_state_table[boundary_vertex])
                if edge_state_table[boundary_vertices[j]]
            end
        end
    end
end
        

end