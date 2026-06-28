use crate::NodeID;

#[derive(Debug)]
pub enum GraphError {
    NodeNotInGraph(NodeID),
    IsolatedNode(NodeID),
    SelfConnection(NodeID),
    MExceeded(NodeID),
}
