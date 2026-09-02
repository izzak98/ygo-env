from pathlib import Path

from ygoenv.paths import cards_db, code_list_path, deck_path, get_repo_root, scripts_path


def test_repo_root_is_ygo_env():
    root = get_repo_root()
    assert (root / "assets" / "deck").is_dir()
    assert (root / "example" / "code_list.txt").is_file()


def test_deck_path_resolves_stem():
    p = deck_path("tear")
    assert p.name == "tear.ydk"
    assert p.parent.name == "deck"


def test_code_list_path():
    assert code_list_path().name == "code_list.txt"


def test_cards_db_default_en():
    assert cards_db("en").parts[-3:] == ("locale", "en", "cards.cdb")


def test_scripts_path_is_ygopro_scripts():
    p = scripts_path()
    assert p.name == "ygopro-scripts"
    assert p.parent.name == "third_party"
