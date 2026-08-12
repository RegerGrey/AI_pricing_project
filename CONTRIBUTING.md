# Contributing to ConjointKit

1. Fork or clone the repository and create a branch.
2. Create a Python 3.11+ environment.
3. Install the project and development tools:

   ```bash
   pip install -e .[dev]
   ```

4. Run checks before opening a pull request:

   ```bash
   pytest
   ruff check .
   ```

5. To work on the interface, start it from the repository root:

   ```bash
   streamlit run app/streamlit_app.py
   ```

Please keep changes small, add a focused test for changed behavior, and avoid including respondent data or files without clear redistribution rights.
