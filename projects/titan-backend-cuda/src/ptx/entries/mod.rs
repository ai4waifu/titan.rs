//! PTX entry builders (atomic instruction sequences).

use super::ast::{ElementwiseOperation, Entry, Identifier};

mod attention;
mod broadcast_add;
mod concat;
mod conv2d;
mod elementwise;
mod gelu;
mod gemm;
mod group_norm;
mod layer_norm;
mod params;
mod prologue;
mod quick_gelu;
mod reduction_sum;
mod regs;
mod resize;
mod silu;
mod slice;
mod softmax;
mod transpose;

impl Entry {
    pub(super) fn elementwise_f32(name: Identifier, operation: ElementwiseOperation) -> Self {
        elementwise::elementwise_f32(name, operation)
    }

    pub(super) fn silu_f32(name: Identifier) -> Self {
        silu::silu_f32(name)
    }

    pub(super) fn quick_gelu_f32(name: Identifier) -> Self {
        quick_gelu::quick_gelu_f32(name)
    }

    pub(super) fn gelu_f32(name: Identifier) -> Self {
        gelu::gelu_f32(name)
    }

    pub(super) fn gemm_f32(name: Identifier) -> Self {
        gemm::gemm_f32(name)
    }

    pub(super) fn conv2d_f32(name: Identifier) -> Self {
        conv2d::conv2d_f32(name)
    }

    pub(super) fn scaled_dot_product_attention_f32(name: Identifier) -> Self {
        attention::scaled_dot_product_attention_f32(name)
    }

    pub(super) fn broadcast_add_f32(name: Identifier) -> Self {
        broadcast_add::broadcast_add_f32(name)
    }

    pub(super) fn softmax_f32(name: Identifier) -> Self {
        softmax::softmax_f32(name)
    }

    pub(super) fn reduction_sum_f32(name: Identifier) -> Self {
        reduction_sum::reduction_sum_f32(name)
    }

    pub(super) fn concat_f32(name: Identifier) -> Self {
        concat::concat_f32(name)
    }

    pub(super) fn transpose_f32(name: Identifier) -> Self {
        transpose::transpose_f32(name)
    }

    pub(super) fn slice_f32(name: Identifier) -> Self {
        slice::slice_f32(name)
    }

    pub(super) fn resize_nearest2d_f32(name: Identifier) -> Self {
        resize::resize_nearest2d_f32(name)
    }

    pub(super) fn layer_norm_f32(name: Identifier) -> Self {
        layer_norm::layer_norm_f32(name)
    }

    pub(super) fn group_norm_f32(name: Identifier) -> Self {
        group_norm::group_norm_f32(name)
    }
}
