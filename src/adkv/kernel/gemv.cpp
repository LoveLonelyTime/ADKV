#include <torch/extension.h>
#include <tuple>

torch::Tensor forward_qk(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor k_new,
    torch::Tensor k_zp_init, torch::Tensor k_sc_init,
    torch::Tensor k_alpha, torch::Tensor k_beta, torch::Tensor k_t,
    int T,
    float scale_min, float scale_max);

torch::Tensor forward_ov(
    torch::Tensor attn,
    torch::Tensor v_cache,
    torch::Tensor v_new,
    torch::Tensor v_zp_init, torch::Tensor v_sc_init,
    torch::Tensor v_alpha, torch::Tensor v_beta, torch::Tensor v_t,
    float scale_min, float scale_max);

torch::Tensor encode_kv(
    torch::Tensor kv_cache,
    torch::Tensor kv_new,
    torch::Tensor kv_zp_init, torch::Tensor kv_sc_init,
    torch::Tensor kv_alpha, torch::Tensor kv_beta, torch::Tensor kv_t,
    float scale_min, float scale_max);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward_qk", torch::wrap_pybind_function(forward_qk), "forward_qk");
    m.def("forward_ov", torch::wrap_pybind_function(forward_ov), "forward_ov");
    m.def("encode_kv", torch::wrap_pybind_function(encode_kv), "encode_kv");
}
