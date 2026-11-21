# ARSA Cross-Engine Integration Guide

**ARSA (Active Ragdoll Simulation Architecture)** provides a deterministic, sub-stepped XPBD + Stable-PD active ragdoll engine with native C ABI (`arsa-c-abi`) and WebAssembly (`arsa-wasm`) bindings.

---

## 1. Building the Libraries

### Native C/C++ Shared & Static Libraries (`arsa-c-abi`)

```bash
# Build optimized release libraries in target/release/
# - Linux:   target/release/libarsa_c_abi.so, libarsa_c_abi.a
# - Windows: target/release/arsa_c_abi.dll, arsa_c_abi.dll.lib, arsa_c_abi.lib
# - macOS:   target/release/libarsa_c_abi.dylib, libarsa_c_abi.a
cargo build --release -p arsa-c-abi
```

The C/C++ header is located at `bindings/c-abi/include/arsa.h`.

### WebAssembly (`arsa-wasm`)

```bash
wasm-pack build bindings/wasm --target web --out-dir ../../playground/src/wasm
```

---

## 2. Dual Integration Modes

ARSA supports two engine integration architectures:

### Mode A: Standalone Simulation Mode (Recommended for Rapid Integration)
ARSA owns the rigid-body solver, ground/obstacle contacts, active balance controller, capture-point stepping, and injury/wound reflexes. Each physics tick:
1. Call `arsa_world_step(world, dt)`.
2. Read the 17 bone world transforms via `arsa_world_get_transforms(world, buf, max_floats)` (7 floats per body: `px, py, pz, qx, qy, qz, qw`).
3. Convert coordinates to your engine's handedness/up-axis and write the transforms to your visual skeleton or mesh bones.
4. Optionally query `arsa_world_get_ragdoll_telemetry(world, tel, 16)` for behavior state, center of mass, capture point, health, and active wound location.

### Mode B: Host-Physics Motor Target Mode (UE5 Chaos / Unity PhysX)
When your host engine owns world collisions and character ragdoll bodies (e.g., Chaos `UPhysicsAsset` or Unity `ArticulationBody` / `ConfigurableJoint`):
1. Step ARSA's behavioral brain alongside your character (`arsa_world_step`).
2. Read per-joint PD motor targets via `arsa_world_get_joint_motor_targets(world, motors, max_floats)`.
3. Each of the 16 articulated joints outputs **8 floats** (`ArsaJointMotorTarget`):
   `qx, qy, qz, qw, stiffness, damping, max_torque, damage_scale`.
4. Feed `(qx, qy, qz, qw)` as the angular drive target orientation and scale your engine's angular motor spring/damping/torque limits by `stiffness * damage_scale`, `damping * damage_scale`, and `max_torque * damage_scale`.

---

## 3. Coordinate System Conversion Table

ARSA uses a **Right-Handed, Y-Up, Meter** coordinate frame (`+X` character left, `+Y` up, `+Z` character forward) with XYZW quaternions.

| Engine | Handedness | Up Axis | Forward Axis | Units | Position `(X, Y, Z)` from ARSA `(x, y, z)` | Quaternion `(QX, QY, QZ, QW)` from ARSA `(qx, qy, qz, qw)` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ARSA** | Right-Handed | `+Y` | `+Z` | Meters | `(x, y, z)` | `(qx, qy, qz, qw)` |
| **Three.js** | Right-Handed | `+Y` | `+Z` | Meters | `(x, y, z)` | `(qx, qy, qz, qw)` |
| **Unity** | Left-Handed | `+Y` | `+Z` | Meters | `(-x, y, z)` | `(qx, -qy, -qz, qw)` |
| **Unreal Engine 5** | Left-Handed | `+Z` | `+X` | Centimeters | `(z * 100, -x * 100, y * 100)` | `(-qz, qx, -qy, qw)` |

---

## 4. Quickstart Snippets

### C / C++

```c
#include "arsa.h"

ArsaWorldHandle *world = arsa_world_create();
arsa_world_spawn_ragdoll_profile(world, 0.0f, 0.0f, 0.0f, 1.78f, 75.0f, 1.0f);

for (int frame = 0; frame < 120; ++frame) {
    arsa_world_step(world, 1.0f / 60.0f);
}

float transforms[ARSA_BONE_COUNT * ARSA_TRANSFORM_FLOATS_PER_BODY];
arsa_world_get_transforms(world, transforms, sizeof(transforms) / sizeof(float));

arsa_world_fire_weapon(world, 0.0f, 1.2f, 2.0f, 0.0f, 0.0f, -1.0f, ARSA_WEAPON_PISTOL);
arsa_world_destroy(world);
```

See `bindings/c-abi/examples/smoke_test.c` for a complete runnable example.

### Unity (UPM Package: `bindings/unity/`)

1. In Unity, open **Window -> Package Manager -> Add package from disk...** and select `bindings/unity/package.json`.
2. Copy the compiled `arsa_c_abi` native library (`arsa_c_abi.dll` / `libarsa_c_abi.so` / `libarsa_c_abi.dylib`) into `Assets/Plugins/<Platform>/`.
3. Attach `ArsaRagdollAgent` to a character root and assign the 17 `Transform` slots in `ArsaBoneId` order:

```csharp
using Arsa;
using UnityEngine;

public class RagdollShooter : MonoBehaviour
{
    public ArsaRagdollAgent agent;

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            agent.FireWeapon(ray.origin, ray.direction, ArsaWeaponKind.Pistol);
        }
    }
}
```

### Unreal Engine 5 (`bindings/unreal/ArsaRagdoll/`)

1. Copy `bindings/unreal/ArsaRagdoll/` into your project's `Plugins/` folder (and copy `arsa.h` + platform binaries into `ThirdParty/arsa/` if building outside this repo).
2. Enable **ARSA Active Ragdoll** in **Edit -> Plugins**.
3. Add `UArsaRagdollComponent` to an Actor and read `BoneTransforms` in `Tick` or Blueprints:

```cpp
// Example C++ usage on an Actor with UArsaRagdollComponent* RagdollComp
RagdollComp->UpdateBodyProfile(2.17f, 135.0f, 1.2f);
RagdollComp->FireWeapon(CameraLoc, CameraForward, EArsaWeaponKind::Shotgun);
EArsaBehaviorState State = RagdollComp->GetBehaviorState();
```

### Web / Three.js (`bindings/wasm/`)

```ts
import init, { ArsaSim } from "./wasm/arsa_wasm.js";

await init();
const sim = new ArsaSim();
sim.spawn_ragdoll_with_profile(0, 0, 0, 1.78, 75.0, 1.0);

function animate() {
  sim.step(1 / 60);
  const transforms = sim.get_transforms(); // Float32Array: 7 floats per body
  const telemetry = sim.get_ragdoll_telemetry(); // Float32Array: 16 floats per ragdoll
  requestAnimationFrame(animate);
}
animate();
```
