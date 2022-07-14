module ClusterTrees

    # import Pkg; Pkg.add("AbstractTrees")
    using AbstractTrees

    if !isdefined(Base, :isnothing)        # Julia 1.0 support
        using AbstractTrees: isnothing
    end

    mutable struct ClusterTreeNode{T}
        data::T
        parent::Union{Nothing,ClusterTreeNode{T}}
        children::Union{Nothing, Set{ClusterTreeNode{T}}}
        clustersize::Int #excludes root
        parity::Bool
        rootnode::Union{Nothing,ClusterTreeNode{T}} #After initializing the rootnode, 

        function ClusterTreeNode{T}(data, parent=nothing, children=nothing, clustersize=0, parity=true, rootnode=nothing) where T
            new{T}(data, parent, children, clustersize, parity, rootnode)
        end
    end

    ClusterTreeNode(data) = ClusterTreeNode{typeof(data)}(data)

    function createnewchild!(parent::ClusterTreeNode, data)
        new_child = typeof(parent)(data, parent)
        if isnothing(parent.children)
            parent.children = Set([new_child])
        else
            push!(parent.children, new_child)
        end
        if isnothing(parent.parent)
            parent.clustersize += 1
            parent.parity = xor(parent.parity, new_child.parity)
            new_child.rootnode = parent
        else
            parent.rootnode.clustersize += 1
            parent.rootnode.parity = xor(parent.rootnode.parity, new_child.parity)
            new_child.rootnode = parent.rootnode
        end
    end

    function pathcompression_findrootnode(current_node,data_for_rootnode)
        # data_for_rootnode will be a tuple where the first value is the added clustersize for the subtree
        # the second value is the parity for the added subtree
        if isnothing(current_node.parent)
            current_node.cluster_size += data_for_rootnode[0]
            current_node.parity = xor(current_node.parity, data_for_rootnode[1])
            return current_node
        else
            rootnode = pathcompression_findrootnode(current_node.rootnode, data_for_rootnode)
            current_node.rootnode = rootnode
        end
    end

    function addchild!(parent::ClusterTreeNode, child::ClusterTreeNode)
        if isnothing(parent.children)
            parent.children = Set([child])
        else
            push!(parent.children, child)
        end
        pathcompression_findrootnode(parent, (child.cluster_size, child.parity))
        child.rootnode = parent.rootnode
        child.parent = parent
    end

    function reversedepth!(node::ClusterTreeNode)
        counter = 0
        current_node = node
        while !isnothing(current_node.parent)
            counter = 1
            current_node = current_node.parent
        end
        return counter
    end


    ## Things we need to define
    # function AbstractTrees.children(node::ClusterTreeNode)
    #     if isnothing(node.children)
    #         ()
    #     else
    #         (node.children...)
    #     end
    # end

    AbstractTrees.nodevalue(n::ClusterTreeNode) = n.data

    AbstractTrees.ParentLinks(::Type{<:ClusterTreeNode}) = StoredParents()

    AbstractTrees.parent(n::ClusterTreeNode) = n.parent

    AbstractTrees.NodeType(::Type{<:ClusterTreeNode{T}}) where {T} = HasNodeType()
    AbstractTrees.nodetype(::Type{<:ClusterTreeNode{T}}) where {T} = ClusterTreeNode{T}
end