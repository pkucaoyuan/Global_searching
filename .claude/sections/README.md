# Claude Code Sections - Modular Documentation

This directory contains modular sections of the CLAUDE.md file to improve maintainability and readability.

## Structure

```
.claude/
├── CLAUDE.md                 # Main project guidelines (core essentials)
└── sections/                 # Detailed sections (referenced from main)
    ├── README.md             # This file
    ├── testing_cicd.md       # Testing and CI/CD guidelines
    ├── debugging.md          # Debugging and error handling
    ├── deployment.md         # Production deployment procedures
    ├── performance.md        # Performance optimization guide
    ├── contribution.md       # Contribution guidelines
    └── configuration.md      # Configuration management
```

## Purpose

The modularization serves several purposes:

1. **Maintainability**: Easier to update individual sections without navigating a large file
2. **Readability**: Main CLAUDE.md stays focused on core principles
3. **Navigation**: Sections can be read independently based on need
4. **Collaboration**: Multiple people can edit different sections simultaneously

## Section Overview

### testing_cicd.md
- Test framework and conventions
- CI/CD workflow (GitHub Actions)
- Test writing standards (AAA pattern)
- Coverage requirements

### debugging.md
- Systematic debugging workflow
- Common training failure diagnosis
- Metrics monitoring checklist
- Error handling strategy

### deployment.md
- Production readiness checklist
- Environment management (dev/staging/prod)
- Checkpoint version control
- Rollback procedures
- SLA definitions

### performance.md
- Profiling workflow
- Performance tuning parameters
- Memory optimization
- Distributed training guide
- Scaling strategies

### contribution.md
- PR workflow and standards
- Branch naming conventions
- Commit message format
- Code review checklist
- Deprecation policy

### configuration.md
- Configuration file structure
- Multi-environment strategy
- Configuration validation
- Parameter documentation
- Version control

## Usage

These sections are referenced from the main CLAUDE.md using links:

```markdown
详见: [.claude/sections/testing_cicd.md](.claude/sections/testing_cicd.md)
```

## Updates

When updating these sections:
1. Keep main CLAUDE.md in sync with section summaries
2. Update cross-references if section structure changes
3. Maintain consistent formatting across all sections
4. Update this README if new sections are added

---

*Last updated: 2026-01-17*
*Documentation version: v1.9.0*
