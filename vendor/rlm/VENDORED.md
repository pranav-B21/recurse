# Vendored: rlm

- **Upstream:** https://github.com/alexzhang13/rlm.git
- **Vendored from commit:** `156fd725411b9cae822f5920a6cbf102a5473baa`
  ("Single llm() failure in llm_batch() should not error out the whole batch (lint)")
- **License:** MIT (see `LICENSE` — kept intact)

This is a modifiable fork, editable-installed into the project venv
(`pip install -e vendor/rlm`). Local modifications are tagged with
`# RECURSE-MOD:` comments at each site and catalogued in `RECURSE-MODS.md`.

To sync with upstream manually: fetch the upstream repo, diff against the
commit above, and re-apply the mods listed in `RECURSE-MODS.md`.

Never `pip install rlms` from PyPI — it shadows this fork.
