using AbstractTrees

if !isdefined(Base, :isnothing)        # Julia 1.0 support
    using AbstractTrees: isnothing
end

mutable struct ClusterTreeNode{T}
    data::T
    parent::Union{Nothing,ClusterTreeNode{T}}
    children::Union{Nothing, Set{ClusterTreeNode{T}}}

    function ClusterTreeNode{T}(data, parent=nothing, children=nothing) where T
        new{T}(data, parent, children)
    end
end

ClusterTreeNode(data) = ClusterTreeNode{typeof(data)}(data)

function addchild!(parent::ClusterTreeNode, data)
    if isnothing(parent.children)
        parent.children = Set([typeof(parent)(data, parent)])
    else
        push!(parent.children, typeof(parent)(data, parent))
    end
end


## Things we need to define
function AbstractTrees.children(node::ClusterTreeNode)
    if isnothing(node.children)
        ()
    else
        (node.children...)
    end
end

AbstractTrees.nodevalue(n::ClusterTreeNode) = n.data

AbstractTrees.ParentLinks(::Type{<:ClusterTreeNode}) = StoredParents()

AbstractTrees.parent(n::ClusterTreeNode) = n.parent

AbstractTrees.NodeType(::Type{<:BinarClusterTreeNodeyNode{T}}) where {T} = HasNodeType()
AbstractTrees.nodetype(::Type{<:ClusterTreeNode{T}}) where {T} = ClusterTreeNode{T}
