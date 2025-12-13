use crate::nodes::Node;

#[derive(Debug)]
pub enum GraphError {
    NodeNotInGraph(Node),
    WouldIsolateNode(Node),
    SelfConnection(Node),
}
