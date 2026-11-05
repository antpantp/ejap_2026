# Manual GitHub upload

1. Review `publication_decisions.md` with the authors and review the included MIT licence and finalize the citation and data-access wording.
2. Sign in to GitHub and create a new repository, for example `prbs-vo2peak-prediction`. Choose **Public** for unrestricted viewing. This can be a new repository under your existing personal account; the working project does not need an existing connection to it.
3. Extract `prbs-vo2max-code.zip`. Upload the contents of its `prbs-vo2max-code` folder, preserving subfolders. Upload the extracted source files, rather than just the ZIP. Include the hidden `.gitignore` and `.python-version` files.
4. Commit the uploaded files. Review the rendered README and confirm that no measurements, model artifacts, or private manuscript files were included.
5. Open the repository URL while signed out to confirm public access.
6. Create a tagged release, such as `v1.0.0`, for the version supplied with the manuscript. Put the repository/release URL in the manuscript's code-availability statement. If the journal requires a DOI or long-term archive, deposit that release with the journal's accepted archive service.

Suggested code-availability wording, after the repository is public:

> The analysis and figure-generation code is publicly available at [repository URL], release [version]. Participant data are not included in the code repository; data availability is described separately in the Data Availability statement.

GitHub instructions: https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository
