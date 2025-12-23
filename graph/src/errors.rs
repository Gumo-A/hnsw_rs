use crate::nodes::Node;

#[derive(Debug)]
pub enum GraphError {
    NodeNotInGraph(Node),
    IsolatedNode(Node),
    SelfConnection(Node),
    MExceeded(Node),
}
