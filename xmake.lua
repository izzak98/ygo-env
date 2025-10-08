add_rules("mode.debug", "mode.release")

-- local xmake package recipes
add_repositories("my-repo repo")

-- deps (system sqlite to avoid extra build)
add_requires(
    "ygopro-core",
    "pybind11 2.13.*", "fmt 10.2.*", "glog 0.6.0",
    "sqlite3 3.43.0+200", {system = true},
    "sqlitecpp 3.2.1",    {system = true},
    "concurrentqueue 1.0.4", "unordered_dense 4.4.*"
)

target("ygopro_ygoenv")
    add_rules("python.module")  -- was: python.library
    add_files("ygoenv/ygoenv/ygopro/*.cpp")
    add_packages("pybind11", "fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "ygopro-core")
    set_languages("c++17")
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
        add_cxxflags("-march=native")
    end
    add_includedirs("ygoenv")
    after_build(function (target)
        local install_target = "$(projectdir)/ygoenv/ygoenv/ygopro"
        os.cp(target:targetfile(), install_target)
        print("Copy target to " .. install_target)
    end)
