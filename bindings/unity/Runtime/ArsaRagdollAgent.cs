using System;
using UnityEngine;

namespace Arsa
{
    /// <summary>
    /// Unity MonoBehaviour driving a 17-bone ARSA active ragdoll simulation.
    /// Place bone Transforms in <see cref="boneTransforms"/> ordered by <see cref="ArsaBoneId"/>.
    /// </summary>
    [DisallowMultipleComponent]
    public class ArsaRagdollAgent : MonoBehaviour
    {
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

        [Header("Skeleton Rig (17 Bones in ArsaBoneId Order)")]
        /// <summary>
        /// 17 bone transforms ordered by <see cref="ArsaBoneId"/>.
        /// Expected as flat sibling children under the agent root (because <c>SyncBoneTransforms</c>
        /// writes root-relative <c>localPosition</c> and <c>localRotation</c>), or synchronized in world space.
        /// </summary>
        public Transform[] boneTransforms = new Transform[(int)ArsaBoneId.Count];

        public ArsaBehaviorState CurrentState { get; private set; } = ArsaBehaviorState.Balanced;
        public float Health { get; private set; } = 100.0f;
        public ArsaWoundRegion ActiveWoundRegion { get; private set; } = ArsaWoundRegion.None;

        private IntPtr _world = IntPtr.Zero;
        private readonly float[] _transformBuffer = new float[(int)ArsaBoneId.Count * ArsaNative.TransformFloatsPerBody];
        private readonly float[] _telemetryBuffer = new float[ArsaNative.TelemetryFloatsPerRagdoll];

        private void Awake()
        {
            _world = ArsaNative.arsa_world_create();
            if (_world == IntPtr.Zero)
            {
                Debug.LogError("[ARSA] Failed to allocate native ArsaWorldHandle.", this);
                return;
            }

            ArsaNative.arsa_world_spawn_ragdoll_profile(
                _world,
                0.0f,
                0.0f,
                0.0f,
                heightMeters,
                massKg,
                muscleRatio);
            ArsaNative.arsa_world_set_active(_world, activeBalance);
            ArsaNative.arsa_world_set_auto_get_up(_world, autoGetUp);
        }

        private void Start()
        {
            if (_world != IntPtr.Zero)
            {
                ArsaNative.arsa_world_set_auto_get_up(_world, autoGetUp);
            }
        }

        private void FixedUpdate()
        {
            if (_world == IntPtr.Zero)
            {
                return;
            }

            ArsaNative.arsa_world_step(_world, Time.fixedDeltaTime);
            SyncBoneTransforms();
            SyncTelemetry();
        }

        private void OnValidate()
        {
            heightMeters = Mathf.Clamp(heightMeters, 1.40f, 2.40f);
            massKg = Mathf.Clamp(massKg, 40.0f, 180.0f);
            muscleRatio = Mathf.Clamp(muscleRatio, 0.25f, 2.0f);

            if (boneTransforms == null || boneTransforms.Length != (int)ArsaBoneId.Count)
            {
                Array.Resize(ref boneTransforms, (int)ArsaBoneId.Count);
            }

            if (_world != IntPtr.Zero)
            {
                ArsaNative.arsa_world_update_all_profiles(_world, heightMeters, massKg, muscleRatio);
                ArsaNative.arsa_world_set_active(_world, activeBalance);
                ArsaNative.arsa_world_set_auto_get_up(_world, autoGetUp);
            }
        }

        private void OnDestroy()
        {
            if (_world != IntPtr.Zero)
            {
                ArsaNative.arsa_world_destroy(_world);
                _world = IntPtr.Zero;
            }
        }

        /// <summary>
        /// Fires a ballistic weapon ray (in Unity world space) into the ragdoll.
        /// Returns the number of ray hits.
        /// </summary>
        public int FireWeapon(Vector3 worldOrigin, Vector3 worldDirection, ArsaWeaponKind weapon)
        {
            if (_world == IntPtr.Zero)
            {
                return 0;
            }

            Vector3 localO = ArsaNative.ToArsaVector(transform.InverseTransformPoint(worldOrigin));
            Vector3 localD = ArsaNative.ToArsaVector(transform.InverseTransformDirection(worldDirection));
            return ArsaNative.arsa_world_fire_weapon(
                _world,
                localO.x,
                localO.y,
                localO.z,
                localD.x,
                localD.y,
                localD.z,
                (uint)weapon);
        }

        /// <summary>
        /// Applies a broad two-handed shove impulse along a Unity world-space ray.
        /// Returns the hit bone index, or -1 if missed.
        /// </summary>
        public int ApplyShove(Vector3 worldOrigin, Vector3 worldDirection, float strength = 300.0f)
        {
            if (_world == IntPtr.Zero)
            {
                return -1;
            }

            Vector3 localO = ArsaNative.ToArsaVector(transform.InverseTransformPoint(worldOrigin));
            Vector3 localD = ArsaNative.ToArsaVector(transform.InverseTransformDirection(worldDirection));
            return ArsaNative.arsa_world_apply_shove(
                _world,
                localO.x,
                localO.y,
                localO.z,
                localD.x,
                localD.y,
                localD.z,
                strength);
        }

        /// <summary>
        /// Applies a sharp localized punch impulse along a Unity world-space ray.
        /// Returns the hit bone index, or -1 if missed.
        /// </summary>
        public int ApplyPunch(Vector3 worldOrigin, Vector3 worldDirection, float strength = 150.0f)
        {
            if (_world == IntPtr.Zero)
            {
                return -1;
            }

            Vector3 localO = ArsaNative.ToArsaVector(transform.InverseTransformPoint(worldOrigin));
            Vector3 localD = ArsaNative.ToArsaVector(transform.InverseTransformDirection(worldDirection));
            return ArsaNative.arsa_world_apply_punch(
                _world,
                localO.x,
                localO.y,
                localO.z,
                localD.x,
                localD.y,
                localD.z,
                strength);
        }

        /// <summary>
        /// Enables or disables automatic procedural get-up from <see cref="ArsaBehaviorState.Downed"/>.
        /// </summary>
        public void SetAutoGetUp(bool enabled)
        {
            autoGetUp = enabled;
            if (_world != IntPtr.Zero)
            {
                ArsaNative.arsa_world_set_auto_get_up(_world, autoGetUp);
            }
        }

        /// <summary>
        /// Sets the earthquake seismic magnitude (<c>0.0..=9.0</c>, <c>0.0</c> disables).
        /// </summary>
        public void SetEarthquake(float magnitude)
        {
            if (_world != IntPtr.Zero)
            {
                ArsaNative.arsa_world_set_earthquake(_world, magnitude);
            }
        }

        /// <summary>
        /// Fires a radial + directional kinetic force wave along a Unity world-space ray.
        /// Returns the number of affected bodies.
        /// </summary>
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

            Vector3 localO = ArsaNative.ToArsaVector(transform.InverseTransformPoint(worldOrigin));
            Vector3 localD = ArsaNative.ToArsaVector(transform.InverseTransformDirection(worldDirection));
            return (int)ArsaNative.arsa_world_apply_force_wave(
                _world,
                localO.x,
                localO.y,
                localO.z,
                localD.x,
                localD.y,
                localD.z,
                radius,
                maxImpulse);
        }

        /// <summary>
        /// Reads current joint PD muscle motor targets (16 joints * 8 floats) for Host-Physics Control Mode.
        /// </summary>
        public int GetJointMotorTargets(float[] outBuffer)
        {
            if (_world == IntPtr.Zero || outBuffer == null || outBuffer.Length == 0)
            {
                return 0;
            }

            return (int)ArsaNative.arsa_world_get_joint_motor_targets(
                _world,
                outBuffer,
                (UIntPtr)outBuffer.Length);
        }

        private void SyncBoneTransforms()
        {
            int floatsRead = (int)ArsaNative.arsa_world_get_transforms(
                _world,
                _transformBuffer,
                (UIntPtr)_transformBuffer.Length);

            int bodyCount = floatsRead / ArsaNative.TransformFloatsPerBody;
            int count = Mathf.Min(bodyCount, boneTransforms.Length);

            for (int i = 0; i < count; i++)
            {
                Transform bone = boneTransforms[i];
                if (bone == null)
                {
                    continue;
                }

                int b = i * ArsaNative.TransformFloatsPerBody;
                bone.localPosition = ArsaNative.ToUnityPosition(
                    _transformBuffer[b + 0],
                    _transformBuffer[b + 1],
                    _transformBuffer[b + 2]);
                bone.localRotation = ArsaNative.ToUnityRotation(
                    _transformBuffer[b + 3],
                    _transformBuffer[b + 4],
                    _transformBuffer[b + 5],
                    _transformBuffer[b + 6]);
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
    }
}
