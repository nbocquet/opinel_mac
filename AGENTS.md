# Repository Guidelines

## Project Structure & Module Organization

This directory is currently empty: no application source, tests, assets, or dependency manifests are present. Introduce directories when the implementation requires them, and document their purpose here. A suggested starting layout is `src/` for application code, `tests/` for automated tests, and `assets/` for static resources. Keep configuration files at the repository root.

## Build, Test, and Development Commands

No build, test, or local development commands are configured yet. When adding the first implementation, provide reproducible setup instructions and record the exact commands in a `README.md`. Include dependency installation, local execution, testing, and any required build step. Do not document commands as available until their configuration exists.

Use RTK by default for supported commands, such as `rtk git status`, `rtk git diff`, and `rtk rg`. Use native commands when RTK is unavailable, unsupported, or hides details needed for verification.

## Coding Style & Naming Conventions

No language or formatter has been selected. Follow the conventions of the chosen language and add formatter or linter configuration alongside the initial source. Use descriptive names, consistent indentation, and small modules with clear responsibilities. Avoid introducing competing formatting tools.

## Testing Guidelines

No testing framework or coverage threshold is configured. Add tests for meaningful behavior and bug fixes as functionality appears. Keep test names descriptive of the behavior being verified, follow the selected framework’s naming convention, and document how to run the suite.

## Commit & Pull Request Guidelines

No Git history is available to establish an existing commit convention. Use concise, imperative commit subjects, such as `Add initial application setup`. Keep changes focused. Pull requests should explain the purpose, summarize validation performed, link related issues when applicable, and include screenshots for visible interface changes.

## Security & Configuration

Never commit credentials or local secrets. When configuration becomes necessary, provide example values and document required environment variables. Exclude generated files and local configuration through `.gitignore`.
