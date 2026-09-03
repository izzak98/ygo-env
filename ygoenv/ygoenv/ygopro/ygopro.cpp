#include "ygoenv/ygopro/ygopro.h"
#include "ygoenv/core/py_envpool.h"
#include <stdexcept>

using YGOProEnvSpec = PyEnvSpec<ygopro::YGOProEnvSpec>;
using YGOProEnvPool = PyEnvPool<ygopro::YGOProEnvPool>;

PYBIND11_MODULE(ygopro_ygoenv, m)
{
  REGISTER(m, YGOProEnvSpec, YGOProEnvPool)
  m.def("init_module", &ygopro::init_module,
        py::arg("db_path"),
        py::arg("code_list_file"),
        py::arg("decks"),
        py::arg("script_dir") = "",
        py::arg("reward_json") = "");
  m.def("load_reward_json", &ygopro::load_reward_json,
        py::arg("reward_json"));
  m.def("set_emb_index_map", &ygopro::set_emb_index_map,
        py::arg("code_to_emb_idx"));

  // Add exception translation
  py::register_exception_translator([](std::exception_ptr p)
                                    {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const std::runtime_error &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (const std::exception &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    } });
}