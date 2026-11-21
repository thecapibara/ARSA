using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Arsa
{
    public enum ArsaBoneId : int
    {
        Pelvis = 0,
        LowerSpine = 1,
        Chest = 2,
        Neck = 3,
        Head = 4,
        LeftUpperArm = 5,
        LeftForearm = 6,
        LeftHand = 7,
        RightUpperArm = 8,
        RightForearm = 9,
        RightHand = 10,
        LeftThigh = 11,
        LeftShin = 12,
        LeftFoot = 13,
        RightThigh = 14,
        RightShin = 15,
        RightFoot = 16,
        Count = 17
    }

    public enum ArsaBehaviorState : uint
    {
        Balanced = 0,
        Stumbling = 1,
        Falling = 2,
        Downed = 3,
        Passive = 4,
        Unconscious = 5,
        Recovering = 6
    }

    public enum ArsaWoundRegion : uint
    {
        None = 0,
        LeftArm = 1,
        RightArm = 2,
        Torso = 3,
        Head = 4,
        LeftLeg = 5,
        RightLeg = 6
    }

    public enum ArsaWeaponKind : uint
    {
        Pistol = 0,
        Shotgun = 1,
        Rifle = 2
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ArsaBodyProfile
    {
        public float heightMeters;
        public float massKg;
        public float muscleRatio;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ArsaTransform
    {
        public float px, py, pz;
        public float qx, qy, qz, qw;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct ArsaJointMotorTarget
    {
        public float qx, qy, qz, qw;
        public float stiffness;
        public float damping;
        public float maxTorque;
        public float damageScale;
    }

    /// <summary>
    /// Raw P/Invoke bindings to the native <c>arsa_c_abi</c> library and
    /// ARSA (Right-Handed Y-up) &lt;-&gt; Unity (Left-Handed Y-up) coordinate conversions.
    /// </summary>
    public static class ArsaNative
    {
        public const string LibName = "arsa_c_abi";
        public const int TransformFloatsPerBody = 7;
        public const int TelemetryFloatsPerRagdoll = 16;
        public const int MotorTargetFloatsPerJoint = 8;

        // ---------------------------------------------------------------------
        // Coordinate System Conversions (ARSA: RH Y-up -> Unity: LH Y-up, mirror X)
        // ---------------------------------------------------------------------

        public static Vector3 ToUnityPosition(float x, float y, float z)
        {
            return new Vector3(-x, y, z);
        }

        public static Quaternion ToUnityRotation(float qx, float qy, float qz, float qw)
        {
            return new Quaternion(qx, -qy, -qz, qw);
        }

        public static Vector3 ToArsaVector(Vector3 unityVec)
        {
            return new Vector3(-unityVec.x, unityVec.y, unityVec.z);
        }

        public static Quaternion ToArsaRotation(Quaternion unityRot)
        {
            return new Quaternion(unityRot.x, -unityRot.y, -unityRot.z, unityRot.w);
        }

        // ---------------------------------------------------------------------
        // Native C ABI Imports
        // ---------------------------------------------------------------------

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr arsa_world_create();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_destroy(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_clear(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_step(IntPtr handle, float dt);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr arsa_world_spawn_ragdoll(IntPtr handle, float x, float y, float z);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr arsa_world_spawn_ragdoll_profile(
            IntPtr handle,
            float x,
            float y,
            float z,
            float heightMeters,
            float massKg,
            float muscleRatio);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_update_all_profiles(
            IntPtr handle,
            float heightMeters,
            float massKg,
            float muscleRatio);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_set_active(
            IntPtr handle,
            [MarshalAs(UnmanagedType.I1)] bool enabled);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_set_auto_get_up(
            IntPtr handle,
            [MarshalAs(UnmanagedType.I1)] bool enabled);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_set_earthquake(
            IntPtr handle,
            float magnitude);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool arsa_world_sync_bone_pose(
            IntPtr handle,
            UIntPtr bodyIdx,
            float px,
            float py,
            float pz,
            float qx,
            float qy,
            float qz,
            float qw,
            float vx,
            float vy,
            float vz,
            float wx,
            float wy,
            float wz);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr arsa_world_spawn_box(
            IntPtr handle,
            float x,
            float y,
            float z,
            float hx,
            float hy,
            float hz,
            float mass,
            float vx,
            float vy,
            float vz);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr arsa_world_add_static_box(
            IntPtr handle,
            float cx,
            float cy,
            float cz,
            float hx,
            float hy,
            float hz,
            float friction,
            float restitution);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int arsa_world_apply_shove(
            IntPtr handle,
            float ox,
            float oy,
            float oz,
            float dx,
            float dy,
            float dz,
            float strength);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int arsa_world_apply_punch(
            IntPtr handle,
            float ox,
            float oy,
            float oz,
            float dx,
            float dy,
            float dz,
            float strength);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int arsa_world_fire_weapon(
            IntPtr handle,
            float ox,
            float oy,
            float oz,
            float dx,
            float dy,
            float dz,
            uint weaponKind);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr arsa_world_apply_force_wave(
            IntPtr handle,
            float ox,
            float oy,
            float oz,
            float dx,
            float dy,
            float dz,
            float radius,
            float maxImpulse);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr arsa_world_body_count(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr arsa_world_joint_count(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr arsa_world_get_transforms(
            IntPtr handle,
            [Out] float[] outBuf,
            UIntPtr maxFloats);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr arsa_world_get_ragdoll_telemetry(
            IntPtr handle,
            [Out] float[] outBuf,
            UIntPtr maxFloats);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr arsa_world_get_joint_motor_targets(
            IntPtr handle,
            [Out] float[] outBuf,
            UIntPtr maxFloats);
    }
}
