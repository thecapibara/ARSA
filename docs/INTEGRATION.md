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

### Mode B: Bidirectional Host-Physics Mode (UE5 Chaos / Unity PhysX)
When your host engine owns world collisions and character ragdoll bodies (e.g., Chaos `UPhysicsAsset` or Unity `Rigidbody` / `ConfigurableJoint`):
1. Before stepping ARSA each physics tick, push each host bone's world pose and linear/angular velocities into ARSA via `arsa_world_sync_bone_pose(world, body_idx, px, py, pz, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz)` (converted to ARSA right-handed Y-up meters; note angular velocity is an axial pseudovector).
2. Run a brain-only controller tick without running ARSA's internal XPBD integrator via `arsa_world_step_brain_only(world, dt)` (or use `arsa_world_step` if reading kinematic targets).
3. Either:
   - **Direct World-Space Torques**: Call `arsa_world_compute_host_joint_torques(world, ragdoll_id, coord_space, out_torques_48)` (`ARSA_SPACE_RH_Y_UP = 0`, `ARSA_SPACE_UNITY_LH_Y_UP = 1`, `ARSA_SPACE_UNREAL_LH_Z_UP = 2`) to obtain 16 joint torques in `N·m` already converted to your engine's coordinate axes, and apply equal-and-opposite torques (`+tau` to child, `-tau` to parent; in UE5 scale `N·m` by `10000.0f` for `AddTorqueInRadians`).
   - **Joint Motor Targets**: Read per-joint PD targets (`8` floats per joint: `qx, qy, qz, qw, stiffness, damping, max_torque, damage_scale`) via `arsa_world_get_joint_motor_targets(world, motors, max_floats)`.

### Retargeting Presets, World-Query Callbacks, Level Geometry & Crowd LOD
- **Skeleton Retargeting (`arsa_world_get_retargeted_bones` / `arsa_world_set_retargeted_target_pose`)**: Converts ARSA's 17-bone arms-down rig directly to/from `ARSA_RETARGET_NATIVE = 0`, `ARSA_RETARGET_MIXAMO_TPOSE = 1`, `ARSA_RETARGET_UNITY_HUMANOID = 2` (LH Y-up, meters, T-pose), and `ARSA_RETARGET_UNREAL_MANNY = 3` (LH Z-up, centimeters, A-pose). Both `ArsaRagdollDriver` (Unity) and `UArsaRagdollComponent` (UE5) use these retargeted transforms directly in `Standalone` mode.
- **Host World-Query Callbacks (`arsa_world_set_query_callbacks`)**: Register `ArsaGroundHeightFn` and `ArsaRaycastWorldFn` callbacks so ARSA queries host terrain and static occlusion (wired to Unity `Physics.RaycastNonAlloc` via `enablePhysXWorldQueries` and UE5 `UWorld::LineTraceSingleByChannel` via `bEnableChaosWorldQueries`). Always unregister with `(world, NULL, NULL, NULL)` on teardown.
- **Native Level Geometry**: Add rotated boxes (`arsa_world_add_static_oriented_box`) or uniform terrain grids (`arsa_world_set_heightfield` / `arsa_world_clear_heightfield`).
- **Crowd-Scaling LOD (`arsa_world_set_lod_focus` / `arsa_world_clear_lod_focus` / `arsa_world_get_ragdoll_lod`)**: Assigns `ARSA_LOD_FULL = 0`, `ARSA_LOD_REDUCED = 1`, or `ARSA_LOD_SLEEP = 2` by distance from a focus point (e.g., active camera).
- **Procedural Get-Up (`arsa_world_set_auto_get_up`)**: Enables or disables automatic procedural get-up from `ARSA_STATE_DOWNED` (`Balanced = 0`, `Stumbling = 1`, `Falling = 2`, `Downed = 3`, `Passive = 4`, `Unconscious = 5`, `Recovering = 6`).
- **Disturbances**: `arsa_world_set_earthquake(world, magnitude)` (`0.0 .. 9.0`) and `arsa_world_set_intoxication(world, level)` (`0.0 .. 1.0`).

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
3. Attach `ArsaRagdollDriver` (or legacy `ArsaRagdollAgent`) to a character root, choose `simulationMode` (`Standalone` or `HostPhysics`) and `retargetPreset` (`UnityHumanoid`), and assign the 17 `Transform` (or `Rigidbody`) slots in `ArsaBoneId` order:

```csharp
using Arsa;
using UnityEngine;

public class RagdollShooter : MonoBehaviour
{
    public ArsaRagdollDriver driver;

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            driver.FireWeapon(ray.origin, ray.direction, ArsaWeaponKind.Pistol);
        }
    }
}
```

### Unreal Engine 5 (`bindings/unreal/ArsaRagdoll/`)

1. Copy `bindings/unreal/ArsaRagdoll/` into your project's `Plugins/` folder (and copy `arsa.h` + platform binaries into `ThirdParty/arsa/` if building outside this repo).
2. Enable **ARSA Active Ragdoll** in **Edit -> Plugins**.
3. Add `UArsaRagdollComponent` to an Actor, configure `SimulationMode` (`Standalone` or `HostPhysics`), `RetargetPreset` (`UnrealManny`), and optional `bEnableChaosWorldQueries`, then read `BoneTransforms` in `Tick` or Blueprints:

```cpp
// Example C++ usage on an Actor with UArsaRagdollComponent* RagdollComp
RagdollComp->UpdateBodyProfile(2.17f, 135.0f, 1.2f);
RagdollComp->SetIntoxication(0.35f);
RagdollComp->SetLodFocus(PlayerCameraLoc, 15.0f, 35.0f);
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
