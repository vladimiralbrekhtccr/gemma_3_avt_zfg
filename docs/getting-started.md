# Getting Started

## Prerequisites

- Python 3.8+ recommended
- `pip` available on your PATH

Install MkDocs:

```bash
pip install mkdocs
```

## Preview Locally

From the project root (where `mkdocs.yml` lives):

```bash
mkdocs serve
```

Then open the URL printed in the terminal (usually http://127.0.0.1:8000/).

## Build Static Site

```bash
mkdocs build
```

The site outputs to the `site/` directory.

## Adding Pages

Create Markdown files under `docs/` (e.g., `docs/usage.md`) and update the `nav` section in `mkdocs.yml` to include them.
