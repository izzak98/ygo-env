# ===== Config =====
SCRIPTS_REPO   := https://github.com/mycard/ygopro-scripts.git
SCRIPTS_DIR    := third_party/ygopro-scripts         # keep inside repo
SCRIPTS_COMMIT := 8e7fde9

# Pinned database commit from your original file
DATABASE_REPO  := https://github.com/mycard/ygopro-database/raw/7b1874301fc1aa52bd60585589f771e372ff52cc/locales
LOCALES        := en zh

# Where the built Python extension lands
YGOENV_PKG_DIR := ygoenv/ygoenv/ygopro
YGOENV_SO_GLOB := $(YGOENV_PKG_DIR)/ygopro_ygoenv*.so

# ===== Phonies =====
.PHONY: all dev assets scripts py_install build_ext clean

# Default: get assets + scripts + install python pkg (no native rebuild)
all: assets scripts py_install

# Dev: also build the C++ extension with xmake
dev: assets scripts py_install build_ext

# ===== Python installs =====
py_install:
	pip install -e ./ygoenv

# ===== Build native extension via xmake =====
# Configure + build once; xmake caches. Use -m release for perf.
build_ext: $(YGOENV_SO_GLOB)

$(YGOENV_SO_GLOB):
	xmake f -m release -y
	xmake b ygopro_ygoenv

# ===== Scripts (card logic) =====
scripts: scripts/link

scripts/link: $(SCRIPTS_DIR)
	@mkdir -p scripts
	ln -sfn "../$(SCRIPTS_DIR)" scripts/script
	@echo "Linked scripts -> scripts/script"

$(SCRIPTS_DIR):
	git clone $(SCRIPTS_REPO) $(SCRIPTS_DIR)
	cd $(SCRIPTS_DIR) && git checkout $(SCRIPTS_COMMIT)

# ===== Assets (card DB + strings) =====
assets: $(LOCALES)

$(LOCALES): % : assets/locale/%/cards.cdb assets/locale/%/strings.conf

assets/locale/en assets/locale/zh:
	mkdir -p $@

assets/locale/en/cards.cdb: assets/locale/en
	wget -nv $(DATABASE_REPO)/en-US/cards.cdb -O $@

assets/locale/en/strings.conf: assets/locale/en
	wget -nv $(DATABASE_REPO)/en-US/strings.conf -O $@

assets/locale/zh/cards.cdb: assets/locale/zh
	wget -nv $(DATABASE_REPO)/zh-CN/cards.cdb -O $@

assets/locale/zh/strings.conf: assets/locale/zh
	wget -nv $(DATABASE_REPO)/zh-CN/strings.conf -O $@

# ===== Cleanup =====
clean:
	rm -rf scripts/script
	rm -rf $(SCRIPTS_DIR)
	rm -rf assets/locale/en assets/locale/zh
