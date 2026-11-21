using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Arsa
{
    public enum ArsaSimulationMode
    {
        Standalone = 0,
        HostPhysics = 1
    }

    /// <summary>
    /// Stage 4 Unity driver supporting both Standalone ARSA XPBD simulation (with retargeted bone
    /// output) and Bidirectional Host-Physics Mode (PhysX rigidbodies driven by ARSA PD torques),
    /// plus optional PhysX world-query callbacks.
    /// </summary>
    [DisallowMultipleComponent]
    public class ArsaRagdollDriver : MonoBehaviour
    {
        [Header("Simulation")]
        [SerializeField]
        public ArsaSimulationMode simulationMode = ArsaSimulationMode.Standalone;

        [SerializeField]
        public ArsaRetargetPreset retargetPreset = ArsaRetargetPreset.UnityHumanoid;

        [SerializeField]
        public bool enablePhysXWorldQueries = false;

        [Header("Anthropometric Profile")]
        [Range(1.40f, 2.40f)]
        public float heightMeters = 1.78f;

        [Range(40.0f, 180.0f)]
        public float massKg = 75.0f;

        [Range(0.25f, 2.0f)]
        public float muscleRatio = 1.0f;

        [Header("Active Control")]
        public bool activeBalance = true;

        [SerializeField]
        public bool autoGetUp = true;

        [Range(0f, 1f)]
        public float intoxication = 0f;

        [Header("Skeleton Rig (17 Bones in ArsaBoneId Order)")]
        public Transform[] boneTransforms = new Transform[(int)ArsaBoneId.Count];

        [SerializeField]
        public Rigidbody[] hostBoneRigidbodies = new Rigidbody[(int)ArsaBoneId.Count];

        public ArsaBehaviorState CurrentState { get; private set; } = ArsaBehaviorState.Balanced;
        public float Health { get; private set; } = 100.0f;
        public ArsaWoundRegion ActiveWoundRegion { get; private set; } = ArsaWoundRegion.None;

        private IntPtr _world = IntPtr.Zero;
        private UIntPtr _ragdollId = UIntPtr.Zero;
        private GCHandle _selfHandle;
        private bool _callbacksRegistered;

        // Pre-allocated per-instance buffers (zero per-frame GC allocations).
        private readonly ArsaBoneTransform[] _retargetedBoneBuffer = new ArsaBoneTransform[(int)ArsaBoneId.Count];
        private readonly float[] _hostTorqueBuffer = new float[ArsaNative.HostTorqueFloatsPerRagdoll];
        private readonly float[] _telemetryBuffer = new float[ArsaNative.TelemetryFloatsPerRagdoll];

        // Static delegate instances prevent delegate garbage collection while registered in native code.
        private static readonly ArsaGroundHeightFn s_groundHeightDelegate = PhysXGroundHeightCallback;
        private static readonly ArsaRaycastWorldFn s_raycastDelegate = PhysXRaycastCallback;
        private static readonly RaycastHit[] s_raycastHits = new RaycastHit[8];

        [ThreadStatic]
        private static float[] s_vec3ScratchA;

        [ThreadStatic]
        private static float[] s_vec3ScratchB;

        [ThreadStatic]
        private static float[] s_normalScratch;

        private void Awake()
        {
            _world = ArsaNative.arsa_world_create();
            if (_world == IntPtr.Zero)
            {
                Debug.LogError("[ARSA] Failed to allocate native ArsaWorldHandle.", this);
                return;
            }

            Vector3 spawnArsa = simulationMode == ArsaSimulationMode.HostPhysics
                ? ArsaNative.ToArsaVector(transform.position)
                : Vector3.zero;

            _ragdollId = ArsaNative.arsa_world_spawn_ragdoll_profile(
                _world,
                spawnArsa.x,
                spawnArsa.y,
                spawnArsa.z,
                heightMeters,
                massKg,
                muscleRatio);

            if (_ragdollId == (UIntPtr)ulong.MaxValue || _ragdollId == (UIntPtr)uint.MaxValue)
            {
                Debug.LogError("[ARSA] Failed to spawn ragdoll in native world.", this);
                ArsaNative.arsa_world_destroy(_world);
                _world = IntPtr.Zero;
                return;
            }

            ArsaNative.arsa_world_set_active(_world, activeBalance);
            ArsaNative.arsa_world_set_auto_get_up(_world, autoGetUp);
            ArsaNative.arsa_world_set_intoxication(_world, intoxication);

            _selfHandle = GCHandle.Alloc(this);
            UpdatePhysXQueryCallbacks();
        }

        private void Start()
        {
            if (_world != IntPtr.Zero)
            {
                ArsaNative.arsa_world_set_auto_get_up(_world, autoGetUp);
                ArsaNative.arsa_world_set_intoxication(_world, intoxication);
                UpdatePhysXQueryCallbacks();
            }
        }

        private void FixedUpdate()
        {
            if (_world == IntPtr.Zero)
            {
                return;
            }

            if (simulationMode == ArsaSimulationMode.HostPhysics)
            {
                StepHostPhysics(Time.fixedDeltaTime);
            }
            else
            {
                ArsaNative.arsa_world_step(_world, Time.fixedDeltaTime);
                SyncRetargetedBoneTransforms();
            }

            SyncTelemetry();
        }

        private void OnValidate()
        {
            heightMeters = Mathf.Clamp(heightMeters, 1.40f, 2.40f);
            massKg = Mathf.Clamp(massKg, 40.0f, 180.0f);
            muscleRatio = Mathf.Clamp(muscleRatio, 0.25f, 2.0f);
            intoxication = Mathf.Clamp01(intoxication);

            if (boneTransforms == null || boneTransforms.Length != (int)ArsaBoneId.Count)
            {
                Array.Resize(ref boneTransforms, (int)ArsaBoneId.Count);
            }

            if (hostBoneRigidbodies == null || hostBoneRigidbodies.Length != (int)ArsaBoneId.Count)
            {
                Array.Resize(ref hostBoneRigidbodies, (int)ArsaBoneId.Count);
            }

            if (_world != IntPtr.Zero)
            {
                ArsaNative.arsa_world_update_all_profiles(_world, heightMeters, massKg, muscleRatio);
                ArsaNative.arsa_world_set_active(_world, activeBalance);
                ArsaNative.arsa_world_set_auto_get_up(_world, autoGetUp);
                ArsaNative.arsa_world_set_intoxication(_world, intoxication);
                UpdatePhysXQueryCallbacks();
            }
        }

        private void OnDestroy()
        {
            if (_world != IntPtr.Zero)
            {
                if (_callbacksRegistered)
                {
                    ArsaNative.arsa_world_set_query_callbacks(_world, IntPtr.Zero, null, null);
                    _callbacksRegistered = false;
                }

                ArsaNative.arsa_world_destroy(_world);
                _world = IntPtr.Zero;
            }

            if (_selfHandle.IsAllocated)
            {
                _selfHandle.Free();
            }
        }

        public void SetIntoxication(float level)
        {
            intoxication = Mathf.Clamp01(level);
            if (_world != IntPtr.Zero)
            {
                ArsaNative.arsa_world_set_intoxication(_world, intoxication);
            }
        }

        public void SetAutoGetUp(bool enabled)
        {
            autoGetUp = enabled;
            if (_world != IntPtr.Zero)
            {
                ArsaNative.arsa_world_set_auto_get_up(_world, autoGetUp);
            }
        }

        public void SetEarthquake(float magnitude)
        {
            if (_world != IntPtr.Zero)
            {
                ArsaNative.arsa_world_set_earthquake(_world, magnitude);
            }
        }

        public void SetLodFocus(Vector3 worldFocus, float reducedDistMeters, float sleepDistMeters)
        {
            if (_world == IntPtr.Zero)
            {
                return;
            }

            Vector3 localFocus = simulationMode == ArsaSimulationMode.HostPhysics
                ? worldFocus
                : transform.InverseTransformPoint(worldFocus);
            Vector3 arsaFocus = ArsaNative.ToArsaVector(localFocus);
            ArsaNative.arsa_world_set_lod_focus(
                _world,
                arsaFocus.x,
                arsaFocus.y,
                arsaFocus.z,
                reducedDistMeters,
                sleepDistMeters);
        }

        public void ClearLodFocus()
        {
            if (_world != IntPtr.Zero)
            {
                ArsaNative.arsa_world_clear_lod_focus(_world);
            }
        }

        public ArsaRagdollLod GetRagdollLod()
        {
            if (_world == IntPtr.Zero)
            {
                return ArsaRagdollLod.Full;
            }

            uint raw = ArsaNative.arsa_world_get_ragdoll_lod(_world, _ragdollId);
            return raw <= (uint)ArsaRagdollLod.Sleep ? (ArsaRagdollLod)raw : ArsaRagdollLod.Full;
        }

        public bool SetRetargetedTargetPose(float[] targetWorldQuats17x4, float blendWeight)
        {
            if (_world == IntPtr.Zero || targetWorldQuats17x4 == null || targetWorldQuats17x4.Length < (int)ArsaBoneId.Count * 4)
            {
                return false;
            }

            return ArsaNative.arsa_world_set_retargeted_target_pose(
                _world,
                _ragdollId,
                (uint)retargetPreset,
                targetWorldQuats17x4,
                blendWeight);
        }

        public bool ClearTargetPose()
        {
            return _world != IntPtr.Zero && ArsaNative.arsa_world_clear_target_pose(_world, _ragdollId);
        }

        public int FireWeapon(Vector3 worldOrigin, Vector3 worldDirection, ArsaWeaponKind weapon)
        {
            if (_world == IntPtr.Zero)
            {
                return 0;
            }

            Vector3 arsaOrigin = WorldPointToArsa(worldOrigin);
            Vector3 arsaDir = WorldDirToArsa(worldDirection);
            return ArsaNative.arsa_world_fire_weapon(
                _world,
                arsaOrigin.x,
                arsaOrigin.y,
                arsaOrigin.z,
                arsaDir.x,
                arsaDir.y,
                arsaDir.z,
                (uint)weapon);
        }

        public int ApplyShove(Vector3 worldOrigin, Vector3 worldDirection, float strength = 300.0f)
        {
            if (_world == IntPtr.Zero)
            {
                return -1;
            }

            Vector3 arsaOrigin = WorldPointToArsa(worldOrigin);
            Vector3 arsaDir = WorldDirToArsa(worldDirection);
            return ArsaNative.arsa_world_apply_shove(
                _world,
                arsaOrigin.x,
                arsaOrigin.y,
                arsaOrigin.z,
                arsaDir.x,
                arsaDir.y,
                arsaDir.z,
                strength);
        }

        public int ApplyPunch(Vector3 worldOrigin, Vector3 worldDirection, float strength = 150.0f)
        {
            if (_world == IntPtr.Zero)
            {
                return -1;
            }

            Vector3 arsaOrigin = WorldPointToArsa(worldOrigin);
            Vector3 arsaDir = WorldDirToArsa(worldDirection);
            return ArsaNative.arsa_world_apply_punch(
                _world,
                arsaOrigin.x,
                arsaOrigin.y,
                arsaOrigin.z,
                arsaDir.x,
                arsaDir.y,
                arsaDir.z,
                strength);
        }

        public int ApplyForceWave(
            Vector3 worldOrigin,
            Vector3 worldDirection,
            float radius = 3.2f,
            float maxImpulse = 42.0f)
        {
            if (_world == IntPtr.Zero)
            {
                return 0;
            }

            Vector3 arsaOrigin = WorldPointToArsa(worldOrigin);
            Vector3 arsaDir = WorldDirToArsa(worldDirection);
            return (int)ArsaNative.arsa_world_apply_force_wave(
                _world,
                arsaOrigin.x,
                arsaOrigin.y,
                arsaOrigin.z,
                arsaDir.x,
                arsaDir.y,
                arsaDir.z,
                radius,
                maxImpulse);
        }

        private void StepHostPhysics(float dt)
        {
            int boneCount = Mathf.Min((int)ArsaBoneId.Count, hostBoneRigidbodies.Length);
            for (int i = 0; i < boneCount; i++)
            {
                Rigidbody rb = hostBoneRigidbodies[i];
                if (rb == null)
                {
                    continue;
                }

                Vector3 p = ArsaNative.ToArsaVector(rb.position);
                Quaternion q = ArsaNative.ToArsaRotation(rb.rotation);
                Vector3 v = ArsaNative.ToArsaVector(rb.velocity);
                Vector3 w = ArsaNative.ToArsaAngularVector(rb.angularVelocity);

                ArsaNative.arsa_world_sync_bone_pose(
                    _world,
                    (UIntPtr)i,
                    p.x, p.y, p.z,
                    q.x, q.y, q.z, q.w,
                    v.x, v.y, v.z,
                    w.x, w.y, w.z);
            }

            ArsaNative.arsa_world_step_brain_only(_world, dt);

            if (!ArsaNative.arsa_world_compute_host_joint_torques(
                    _world,
                    _ragdollId,
                    (uint)ArsaCoordinateSpace.UnityLhYUp,
                    _hostTorqueBuffer))
            {
                return;
            }

            for (int j = 0; j < ArsaNative.JointCount; j++)
            {
                (ArsaBoneId parentId, ArsaBoneId childId) = ArsaNative.JointBonePairs[j];
                int parentIdx = (int)parentId;
                int childIdx = (int)childId;

                int tBase = j * 3;
                Vector3 torque = new Vector3(
                    _hostTorqueBuffer[tBase + 0],
                    _hostTorqueBuffer[tBase + 1],
                    _hostTorqueBuffer[tBase + 2]);

                if (torque.sqrMagnitude < 1e-12f)
                {
                    continue;
                }

                if (childIdx < hostBoneRigidbodies.Length)
                {
                    Rigidbody childRb = hostBoneRigidbodies[childIdx];
                    if (childRb != null && !childRb.isKinematic)
                    {
                        childRb.AddTorque(torque, ForceMode.Force);
                    }
                }

                if (parentIdx < hostBoneRigidbodies.Length)
                {
                    Rigidbody parentRb = hostBoneRigidbodies[parentIdx];
                    if (parentRb != null && !parentRb.isKinematic)
                    {
                        parentRb.AddTorque(-torque, ForceMode.Force);
                    }
                }
            }
        }

        private void SyncRetargetedBoneTransforms()
        {
            if (!ArsaNative.arsa_world_get_retargeted_bones(
                    _world,
                    _ragdollId,
                    (uint)retargetPreset,
                    _retargetedBoneBuffer))
            {
                return;
            }

            int count = Mathf.Min((int)ArsaBoneId.Count, boneTransforms.Length);
            for (int i = 0; i < count; i++)
            {
                Transform bone = boneTransforms[i];
                if (bone == null)
                {
                    continue;
                }

                ArsaBoneTransform b = _retargetedBoneBuffer[i];
                switch (retargetPreset)
                {
                    case ArsaRetargetPreset.UnityHumanoid:
                        bone.localPosition = new Vector3(b.px, b.py, b.pz);
                        bone.localRotation = new Quaternion(b.qx, b.qy, b.qz, b.qw);
                        break;
                    case ArsaRetargetPreset.UnrealManny:
                        bone.localPosition = ArsaNative.ToUnityPosition(-b.py * 0.01f, b.pz * 0.01f, b.px * 0.01f);
                        bone.localRotation = ArsaNative.ToUnityRotation(b.qy, -b.qz, -b.qx, b.qw);
                        break;
                    default:
                        bone.localPosition = ArsaNative.ToUnityPosition(b.px, b.py, b.pz);
                        bone.localRotation = ArsaNative.ToUnityRotation(b.qx, b.qy, b.qz, b.qw);
                        break;
                }
            }
        }

        private void SyncTelemetry()
        {
            int read = (int)ArsaNative.arsa_world_get_ragdoll_telemetry(
                _world,
                _telemetryBuffer,
                (UIntPtr)_telemetryBuffer.Length);

            if (read >= ArsaNative.TelemetryFloatsPerRagdoll)
            {
                CurrentState = (ArsaBehaviorState)Mathf.RoundToInt(_telemetryBuffer[7]);
                Health = _telemetryBuffer[11];
                ActiveWoundRegion = (ArsaWoundRegion)Mathf.RoundToInt(_telemetryBuffer[12]);
            }
        }

        private Vector3 WorldPointToArsa(Vector3 worldPoint)
        {
            Vector3 p = simulationMode == ArsaSimulationMode.HostPhysics
                ? worldPoint
                : transform.InverseTransformPoint(worldPoint);
            return ArsaNative.ToArsaVector(p);
        }

        private Vector3 WorldDirToArsa(Vector3 worldDir)
        {
            Vector3 d = simulationMode == ArsaSimulationMode.HostPhysics
                ? worldDir
                : transform.InverseTransformDirection(worldDir);
            return ArsaNative.ToArsaVector(d);
        }

        private void UpdatePhysXQueryCallbacks()
        {
            if (_world == IntPtr.Zero)
            {
                return;
            }

            if (enablePhysXWorldQueries && !_callbacksRegistered && _selfHandle.IsAllocated)
            {
                ArsaNative.arsa_world_set_query_callbacks(
                    _world,
                    GCHandle.ToIntPtr(_selfHandle),
                    s_groundHeightDelegate,
                    s_raycastDelegate);
                _callbacksRegistered = true;
            }
            else if (!enablePhysXWorldQueries && _callbacksRegistered)
            {
                ArsaNative.arsa_world_set_query_callbacks(_world, IntPtr.Zero, null, null);
                _callbacksRegistered = false;
            }
        }

        private bool IsOwnRigidbody(Rigidbody rb)
        {
            if (rb == null || hostBoneRigidbodies == null)
            {
                return false;
            }

            for (int i = 0; i < hostBoneRigidbodies.Length; i++)
            {
                if (hostBoneRigidbodies[i] == rb)
                {
                    return true;
                }
            }

            return false;
        }

        [AOT.MonoPInvokeCallback(typeof(ArsaGroundHeightFn))]
        private static bool PhysXGroundHeightCallback(
            IntPtr userData,
            float x,
            float z,
            float maxY,
            out float outY,
            IntPtr outNormal3)
        {
            outY = 0f;
            ArsaRagdollDriver driver = TryGetDriver(userData);

            Vector3 localTop = ArsaNative.ToUnityPosition(x, maxY, z);
            Vector3 worldTop = localTop;
            Vector3 worldDown = Vector3.down;

            if (driver != null && driver.simulationMode == ArsaSimulationMode.Standalone)
            {
                worldTop = driver.transform.TransformPoint(localTop);
                worldDown = driver.transform.TransformDirection(Vector3.down).normalized;
            }

            const float maxTraceDistance = 50.0f;
            int hitCount = Physics.RaycastNonAlloc(
                worldTop,
                worldDown,
                s_raycastHits,
                maxTraceDistance,
                Physics.DefaultRaycastLayers,
                QueryTriggerInteraction.Ignore);

            if (!TryGetClosestExternalHit(driver, hitCount, out RaycastHit bestHit))
            {
                return false;
            }

            Vector3 hitUnity = bestHit.point;
            Vector3 normalUnity = bestHit.normal;
            if (driver != null && driver.simulationMode == ArsaSimulationMode.Standalone)
            {
                hitUnity = driver.transform.InverseTransformPoint(hitUnity);
                normalUnity = driver.transform.InverseTransformDirection(normalUnity).normalized;
            }

            Vector3 hitArsa = ArsaNative.ToArsaVector(hitUnity);
            Vector3 normalArsa = ArsaNative.ToArsaVector(normalUnity);
            outY = hitArsa.y;
            WriteNormal3(outNormal3, normalArsa);
            return true;
        }

        [AOT.MonoPInvokeCallback(typeof(ArsaRaycastWorldFn))]
        private static bool PhysXRaycastCallback(
            IntPtr userData,
            IntPtr origin3,
            IntPtr dir3,
            float maxDist,
            out float outDist,
            IntPtr outNormal3)
        {
            outDist = 0f;
            if (origin3 == IntPtr.Zero || dir3 == IntPtr.Zero || maxDist <= 0f)
            {
                return false;
            }

            float[] oBuf = s_vec3ScratchA ?? (s_vec3ScratchA = new float[3]);
            float[] dBuf = s_vec3ScratchB ?? (s_vec3ScratchB = new float[3]);
            Marshal.Copy(origin3, oBuf, 0, 3);
            Marshal.Copy(dir3, dBuf, 0, 3);

            ArsaRagdollDriver driver = TryGetDriver(userData);
            Vector3 unityOrigin = ArsaNative.ToUnityPosition(oBuf[0], oBuf[1], oBuf[2]);
            Vector3 unityDir = ArsaNative.ToUnityPosition(dBuf[0], dBuf[1], dBuf[2]);

            if (driver != null && driver.simulationMode == ArsaSimulationMode.Standalone)
            {
                unityOrigin = driver.transform.TransformPoint(unityOrigin);
                unityDir = driver.transform.TransformDirection(unityDir);
            }

            float dirMag = unityDir.magnitude;
            if (dirMag < 1e-6f)
            {
                return false;
            }

            int hitCount = Physics.RaycastNonAlloc(
                unityOrigin,
                unityDir / dirMag,
                s_raycastHits,
                maxDist,
                Physics.DefaultRaycastLayers,
                QueryTriggerInteraction.Ignore);

            if (!TryGetClosestExternalHit(driver, hitCount, out RaycastHit bestHit))
            {
                return false;
            }

            Vector3 normalUnity = bestHit.normal;
            if (driver != null && driver.simulationMode == ArsaSimulationMode.Standalone)
            {
                normalUnity = driver.transform.InverseTransformDirection(normalUnity).normalized;
            }

            outDist = bestHit.distance;
            WriteNormal3(outNormal3, ArsaNative.ToArsaVector(normalUnity));
            return true;
        }

        private static ArsaRagdollDriver TryGetDriver(IntPtr userData)
        {
            if (userData == IntPtr.Zero)
            {
                return null;
            }

            GCHandle handle = GCHandle.FromIntPtr(userData);
            return handle.IsAllocated ? handle.Target as ArsaRagdollDriver : null;
        }

        private static bool TryGetClosestExternalHit(
            ArsaRagdollDriver driver,
            int hitCount,
            out RaycastHit bestHit)
        {
            bestHit = default;
            float bestDist = float.MaxValue;
            bool found = false;

            for (int i = 0; i < hitCount; i++)
            {
                RaycastHit candidate = s_raycastHits[i];
                if (driver != null && driver.IsOwnRigidbody(candidate.rigidbody))
                {
                    continue;
                }

                if (candidate.distance < bestDist)
                {
                    bestDist = candidate.distance;
                    bestHit = candidate;
                    found = true;
                }
            }

            return found;
        }

        private static void WriteNormal3(IntPtr outNormal3, Vector3 normalArsa)
        {
            if (outNormal3 == IntPtr.Zero)
            {
                return;
            }

            float[] nBuf = s_normalScratch ?? (s_normalScratch = new float[3]);
            nBuf[0] = normalArsa.x;
            nBuf[1] = normalArsa.y;
            nBuf[2] = normalArsa.z;
            Marshal.Copy(nBuf, 0, outNormal3, 3);
        }
    }
}
