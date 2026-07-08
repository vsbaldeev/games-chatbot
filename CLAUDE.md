# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core principles

Ask, don't assume. If something is unclear, ask before writing a single line. Never make silent assumptions about intent, architecture, or requirements.

Flag uncertainty explicitly. If you are not confident about an approach or technical detail, say so before proceeding. Confidence without certainty causes more damage than admitting a gap.

If what we are trying to do is similar to settled science or industry practice, let me know. We don’t have to reinvent the wheel.

I'm always open to ideas on better ways to do things. Please don't hesitate to suggest a better way, or one that has long lasting impact over a tactical change. (as a few examples)

## Design principles

Apply these principles when writing or reviewing code:

**SOLID**
- **Single Responsibility**: each class or module should have one reason to change.
- **Open/Closed**: open for extension, closed for modification — add behaviour by adding code, not editing existing code.
- **Liskov Substitution**: subtypes must be substitutable for their base type without breaking correctness.
- **Interface Segregation**: prefer narrow, focused interfaces over fat ones; don't force clients to depend on methods they don't use.
- **Dependency Inversion**: depend on abstractions, not concretions; inject dependencies rather than hard-coding them.

**General**
- **Composition over inheritance**: build behaviour by combining small objects rather than deep class hierarchies.
- **Law of Demeter**: a method should only talk to its immediate collaborators — avoid chaining through internals (`a.b.c.do()` is a smell).
- **DRY**: every piece of knowledge should have one authoritative location, but don't force an abstraction just to avoid duplication.
- **YAGNI**: don't build for hypothetical future requirements; extend when the need is real.
- **KISS**: the simplest solution that works is usually correct; complexity is a liability.
- **Fail fast**: surface errors early and loudly rather than silently propagating bad state.
- **Separation of concerns**: keep I/O, business logic, and presentation in separate layers.
- **Explicit over implicit**: behaviour should be obvious from reading the code, not hidden in magic or global state.
- **Principle of least surprise**: code should behave the way a reader expects.


## Skills

Always use the `agentic-engineering` skill when designing or implementing agentic features.

Always use the `python-patterns` skill when writing or reviewing Python code.

Always use the `python-testing` skill when writing tests.

## Docstrings

Always add Google-style docstrings to all functions, methods, and classes.

## Documentation policy

After every implementation that changes architecture, adds a feature, or modifies behaviour, update the relevant sections in any affected `README.md` files to reflect the new state. Keep descriptions accurate — stale documentation is worse than none.
