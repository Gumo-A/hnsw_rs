pub trait Serializer {
    fn serialize(&self) -> Vec<u8>;
    fn deserialize(data: Vec<u8>) -> Self;

    /// The number of bytes in the serialized string.
    fn size(&self) -> usize;
}
