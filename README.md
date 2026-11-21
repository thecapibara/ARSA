# ARSA — Active Ragdoll Simulation Architecture

**ARSA** is a cross-engine, physically-based **Active Ragdoll Engine** written in Rust, inspired by NaturalMotion's *Euphoria* (GTA IV). It compiles to:
- **WebAssembly (`arsa-wasm`)** for the real-time Three.js 3D Playground & Showcase.
- **Native C ABI (`arsa-c-abi` / `arsa.h`)** for **Unreal Engine 5**, **Unity**, and **Godot**.

## Architecture
- `core/` (`arsa-core`) — Anthropometric 19-bone humanoid rig, sub-stepped XPBD rigid-body & articulated-joint solver, torque-limited Stable-PD muscles, and procedural behavioral brain (balance, capture-point stepping, reach-for-wound, impact bracing).
- `bindings/wasm/` (`arsa-wasm`) — WebAssembly bindings for the Three.js showcase.
- `bindings/c-abi/` (`arsa-c-abi`) — Engine-agnostic `extern "C"` SDK for UE5, Unity, and Godot.
- `playground/` — Three.js + TypeScript interactive physics test chamber.

## Quickstart
```bash
# Run core physics & unit tests
cargo test

# Build WASM package for the Three.js playground
wasm-pack build bindings/wasm --target web --out-dir ../../playground/src/wasm

# Run the Three.js playground
cd playground && npm install && npm run dev
```
