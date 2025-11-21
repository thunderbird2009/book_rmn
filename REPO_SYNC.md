# Repository Synchronization Guide

This guide documents the workflow for managing content synchronization between the private "Ground Truth" repository and the public "Deployment" repository.

## Repositories

*   **Private Repo (Source of Truth)**: `..\book_rmn_priv`
    *   Contains all raw content, drafts, and private notes.
    *   This is where you should do your writing and editing.
*   **Public Repo (Deployment)**: `.` (Current Directory: `book_rmn`)
    *   Contains the published version of the book.
    *   Includes website configuration (`_config.yml`, `_includes/`, `_scripts/`).
    *   Deploys to GitHub Pages via GitHub Actions.

---

## Workflow 1: Publishing New Content (Private -> Public)

**When:** You have finished writing a chapter or making edits in the private repo and want to publish them to the website.

1.  **In Private Repo (`book_rmn_priv`)**:
    *   Commit your changes.
    ```powershell
    git add .
    git commit -m "Update Chapter X"
    ```

2.  **In Public Repo (`book_rmn`)**:
    *   Ensure the private repo is added as a remote (one-time setup):
        ```powershell
        git remote add upstream ..\book_rmn_priv
        ```
    *   Fetch the latest changes:
        ```powershell
        git fetch upstream
        ```
    *   **Option A: Sync Specific Files (Recommended)**
        *   This avoids merging unrelated history or private files.
        ```powershell
        git checkout upstream/main -- ch5_multi_tower_scoring_model.md ap5_embeddings.md
        ```
    *   **Option B: Merge All Changes**
        *   Use this only if you want to mirror everything.
        ```powershell
        git merge upstream/main --allow-unrelated-histories
        ```

3.  **Push to GitHub**:
    ```powershell
    git add .
    git commit -m "Publish updates from private repo"
    git push origin main
    ```
    *The GitHub Actions workflow will automatically handle LaTeX conversion and front matter injection.*

---

## Workflow 2: Syncing Fixes Back (Public -> Private)

**When:** You made formatting fixes (e.g., LaTeX syntax, typos) directly in the public repo to fix website rendering, and need to update the Ground Truth.

1.  **In Private Repo (`book_rmn_priv`)**:
    *   Ensure the public repo is added as a remote (one-time setup):
        ```powershell
        git remote add public ..\book_rmn
        ```
    *   Fetch the latest fixes:
        ```powershell
        git fetch public
        ```
    *   Checkout the fixed files:
        ```powershell
        git checkout public/main -- ch5_multi_tower_scoring_model.md ap5_embeddings.md
        ```
    *   Commit the fixes:
        ```powershell
        git status
        git commit -m "Sync formatting fixes from public repo"
        ```

---

## LaTeX Formatting Standards

To ensure compatibility with both VS Code Preview and the GitHub Pages (Kramdown) renderer:

1.  **Inline Math**: Use standard LaTeX syntax `$ ... $`.
    *   *Do not* use `\sb` or backticks `` `...` ``.
    *   The build script `_scripts/convert_math.py` automatically converts this to `$$ ... $$` for the website.
2.  **Conditional Probability**: Use `\mid` instead of `|`.
    *   Correct: `$P(A \mid B)$`
    *   Incorrect: `$P(A | B)$` (Breaks Markdown tables).
3.  **Underscores in Text**: Escape them as `\_`.
    *   Correct: `my\_variable`
    *   Incorrect: `my_variable` (Interpreted as italics).
