include("ClusterTrees.jl")
using .ClusterTrees

root = ClusterTrees.ClusterTreeNode(Dict("id"=>0, "coord"=>[1, 1, 1]))
ClusterTrees.createnewchild!(root, Dict("id"=>1, "coord"=>[1,1,2]))
print(root)
