# ===== Config =====
SCRIPTS_REPO   := https://github.com/mycard/ygopro-scripts.git
SCRIPTS_DIR    := third_party/ygopro-scripts
# keep inside repo

# Always pull the latest database from the default branch (HEAD)
DATABASE_REPO  := https://raw.githubusercontent.com/mycard/ygopro-database/HEAD/locales
LOCALES        := en zh

# Where the built Python extension lands
YGOENV_PKG_DIR := ygoenv/ygoenv/ygopro
YGOENV_SO_GLOB := $(YGOENV_PKG_DIR)/ygopro_ygoenv*.so

# ===== Helper: fetch if newer =====
define FETCH_IF_NEWER
	@mkdir -p $(dir $@)
	curl -LsSf -z "$@" -o "$@" "$(1)"
	@echo "Fetched/kept up-to-date: $@"
endef

# ===== Phonies =====
.PHONY: all dev assets scripts py_install build_ext clean scripts/update scripts/link

# Default: get assets + scripts + install python pkg (no native rebuild)
all: assets scripts py_install

# Dev: also build the C++ extension with xmake
dev: assets scripts py_install build_ext

# ===== Python installs =====
py_install:
	pip install -e ./ygoenv

# ===== Build native extension via xmake =====
build_ext: $(YGOENV_SO_GLOB)

$(YGOENV_SO_GLOB):
	xmake f -m release -y
	xmake b ygopro_ygoenv

# ===== Scripts (card logic) =====
scripts: scripts/update scripts/link

# Always update/clone to the latest on the default branch
scripts/update:
	@# If SCRIPTS_DIR is a git repo, fast-forward to the default branch
	@if git -C "$(SCRIPTS_DIR)" rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		echo "Updating scripts repository..."; \
		default_branch=$$(git -C "$(SCRIPTS_DIR)" remote show origin | sed -n '/HEAD branch/s/.*: //p'); \
		if [ -z "$$default_branch" ]; then default_branch=main; fi; \
		git -C "$(SCRIPTS_DIR)" fetch origin "$$default_branch" --tags --prune; \
		git -C "$(SCRIPTS_DIR)" checkout -q "$$default_branch"; \
		git -C "$(SCRIPTS_DIR)" pull --ff-only origin "$$default_branch"; \
	else \
		echo "Recreating scripts directory (not a git repo or broken repo)..."; \
		rm -rf "$(SCRIPTS_DIR)"; \
		mkdir -p "$$(dirname "$(SCRIPTS_DIR)")"; \
		echo "Cloning scripts repository fresh..."; \
		git clone --depth=1 "$(SCRIPTS_REPO)" "$(SCRIPTS_DIR)"; \
	fi


scripts/link:
	@mkdir -p scripts
	ln -sfn "../$(SCRIPTS_DIR)" scripts/script
	@echo "Linked scripts -> scripts/script"

# ===== Assets (card DB + strings) =====
assets: $(LOCALES)

$(LOCALES): % : assets/locale/%/cards.cdb assets/locale/%/strings.conf

assets/locale/en assets/locale/zh:
	mkdir -p $@

assets/locale/en/cards.cdb: assets/locale/en
	$(call FETCH_IF_NEWER,$(DATABASE_REPO)/en-US/cards.cdb)

assets/locale/en/strings.conf: assets/locale/en
	$(call FETCH_IF_NEWER,$(DATABASE_REPO)/en-US/strings.conf)

assets/locale/zh/cards.cdb: assets/locale/zh
	$(call FETCH_IF_NEWER,$(DATABASE_REPO)/zh-CN/cards.cdb)

assets/locale/zh/strings.conf: assets/locale/zh
	$(call FETCH_IF_NEWER,$(DATABASE_REPO)/zh-CN/strings.conf)

# ===== Cleanup =====
clean:
	rm -rf scripts/script
	rm -rf $(SCRIPTS_DIR)
	rm -rf assets/locale/en assets/locale/zh
