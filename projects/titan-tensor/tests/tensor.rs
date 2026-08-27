use titan_tensor::Element;
use titan_types::DType;

#[test]
fn element_protocol_is_stable() {
    assert_eq!(<f32 as Element>::DTYPE, DType::F32);
}
