# Releases and DOI

Each release is a git tag, a GitHub release and a Zenodo archive with a DOI (Zenodo's
GitHub integration copies the tagged repository and mints the DOI). Zenodo reads
`.zenodo.json` (title, creators, description, licence); `CITATION.cff` gives the same
for GitHub's "Cite this repository".

## Once

1. Sign in at https://zenodo.org with GitHub ("Log in with GitHub").
2. Account menu, **GitHub**: find `sahilprg/gnss-interference` and switch it **On**.

## Each release

1. Version in `pyproject.toml`, `src/gnssguard/__init__.py` and `CITATION.cff`
   (`version`, `date-released`); an entry in `CHANGELOG.md`.
2. Tests pass; `gnssguard benchmark` regenerated `results/` if any input changed.
3. Tag and push: `git tag -a v0.2.0 -m "v0.2.0" && git push origin v0.2.0`.
4. GitHub, **Releases, Draft a new release**, tag `v0.2.0`, notes from `CHANGELOG.md`,
   **Publish release**. Zenodo archives it within minutes.
5. Zenodo, the new record: copy the DOI. Add it to `CITATION.cff` (`doi:`) and as a
   badge in `README.md`, commit. The concept DOI (all versions) stays the same across
   releases; the version DOI is new each time.

## What a release fixes

0.2.0 is archived before the sealed test sample is drawn (CHANGELOG.md): the model
hash, labels, baseline and questions in it are the ones the sealed test is scored with.
