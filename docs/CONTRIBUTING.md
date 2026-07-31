## Contributing Standards

To maintain a consistent, professional, and maintainable codebase, all future contributions should follow the standards established during Version 1.0.

### Code Formatting

- Follow the project's established formatting style.
- Maintain consistent spacing, indentation, and line wrapping.
- Prefer readability over compactness.
- Keep functions focused on a single responsibility.

### Naming Conventions

- Use descriptive function and variable names.
- Follow existing naming patterns throughout the project.
- Avoid abbreviations unless they are widely understood.
- Maintain consistency across analytics, visualization, and testing modules.

### Adding New Analytics Modules

When introducing new analytical functionality:

- Place calculations inside the `analytics/` package.
- Keep business logic separate from visualization.
- Expose reusable functions that can be consumed by charts, dashboards, and future interfaces.
- Avoid duplicating calculations already provided elsewhere in the project.

### Testing Expectations

Every new feature should include appropriate automated tests whenever practical.

Tests should verify:

- Correct calculations
- Expected edge-case behavior
- Input validation
- Visualization behavior (where applicable)
- Dashboard integration (if applicable)

Features should not be considered complete until corresponding tests pass.

### Documentation Requirements

Significant additions should be reflected in the project documentation.

Update the appropriate documents when necessary:

- `Development_Log.md`
- `VERSION_HISTORY.md`
- `TECHNICAL_ARCHITECTURE.md`
- `USER_GUIDE.md`

Documentation should evolve alongside the codebase.

### Git Commit Style

Use clear, descriptive commit messages that explain the purpose of the change.

Examples:

- `Add rolling volatility dashboard panel`
- `Refactor forecast styling helpers`
- `Improve export validation tests`
- `Fix dashboard panel population bug`

Avoid vague commit messages such as:

- `Updates`
- `Fix stuff`
- `Changes`

### Project Philosophy

The project follows several guiding principles that should remain consistent as development continues:

- **Analytics owns the mathematics.**
- **Visualization owns presentation.**
- **Interfaces own user interaction.**
- **Calculations should exist only once.**
- **Shared functionality should be centralized whenever practical.**
- **Extensive automated testing is part of feature development, not an afterthought.**
- **Favor modular, reusable components over duplicated logic.**
- **Design new functionality so future desktop, web, and API interfaces can reuse the same underlying analytics without modification.**