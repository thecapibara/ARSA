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

    public enum ArsaRetargetPreset : uint
    {
        Native = 0,
        MixamoTPose = 1,
        UnityHumanoid = 2,
        UnrealManny = 3
    }

    public enum ArsaCoordinateSpace : uint
    {
        RhYUp = 0,
        UnityLhYUp = 1,
        UnrealLhZUp = 2
    }

    public enum ArsaRagdollLod : uint
    {
        Full = 0,
        Reduced = 1,
        Sleep = 2
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
    public struct ArsaBoneTransform
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

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public delegate bool ArsaGroundHeightFn(
        IntPtr userData,
        float x,
        float z,
        float maxY,
        out float outY,
        IntPtr outNormal3);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    [return: MarshalAs(UnmanagedType.I1)]
    public delegate bool ArsaRaycastWorldFn(
        IntPtr userData,
        IntPtr origin3,
        IntPtr dir3,
        float maxDist,
        out float outDist,
        IntPtr outNormal3);

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
        public const int JointCount = 16;
        public const int HostTorqueFloatsPerRagdoll = JointCount * 3;

        /// <summary>
        /// Parent-to-child bone pairs matching <c>JOINT_BONE_PAIRS</c> in <c>arsa-core</c> (0..16).
        /// </summary>
        public static readonly (ArsaBoneId parent, ArsaBoneId child)[] JointBonePairs =
        {
            (ArsaBoneId.Pelvis, ArsaBoneId.LowerSpine),
            (ArsaBoneId.LowerSpine, ArsaBoneId.Chest),
            (ArsaBoneId.Chest, ArsaBoneId.Neck),
            (ArsaBoneId.Neck, ArsaBoneId.Head),
            (ArsaBoneId.Chest, ArsaBoneId.LeftUpperArm),
            (ArsaBoneId.LeftUpperArm, ArsaBoneId.LeftForearm),
            (ArsaBoneId.LeftForearm, ArsaBoneId.LeftHand),
            (ArsaBoneId.Chest, ArsaBoneId.RightUpperArm),
            (ArsaBoneId.RightUpperArm, ArsaBoneId.RightForearm),
            (ArsaBoneId.RightForearm, ArsaBoneId.RightHand),
            (ArsaBoneId.Pelvis, ArsaBoneId.LeftThigh),
            (ArsaBoneId.LeftThigh, ArsaBoneId.LeftShin),
            (ArsaBoneId.LeftShin, ArsaBoneId.LeftFoot),
            (ArsaBoneId.Pelvis, ArsaBoneId.RightThigh),
            (ArsaBoneId.RightThigh, ArsaBoneId.RightShin),
            (ArsaBoneId.RightShin, ArsaBoneId.RightFoot)
        };

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

        public static Vector3 ToArsaAngularVector(Vector3 unityAngularVec)
        {
            return new Vector3(unityAngularVec.x, -unityAngularVec.y, -unityAngularVec.z);
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
        public static extern void arsa_world_step_brain_only(IntPtr handle, float dt);

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
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool arsa_world_set_target_pose(
            IntPtr handle,
            UIntPtr ragdollId,
            [In] float[] quatsXyzw,
            UIntPtr numJoints,
            float blendWeight);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool arsa_world_clear_target_pose(
            IntPtr handle,
            UIntPtr ragdollId);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_set_earthquake(
            IntPtr handle,
            float magnitude);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_set_intoxication(
            IntPtr handle,
            float level);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_set_lod_focus(
            IntPtr handle,
            float x,
            float y,
            float z,
            float reducedDist,
            float sleepDist);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_clear_lod_focus(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint arsa_world_get_ragdoll_lod(
            IntPtr handle,
            UIntPtr ragdollId);

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
        public static extern UIntPtr arsa_world_add_static_oriented_box(
            IntPtr handle,
            float cx,
            float cy,
            float cz,
            float hx,
            float hy,
            float hz,
            float qx,
            float qy,
            float qz,
            float qw,
            float friction,
            float restitution);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool arsa_world_set_heightfield(
            IntPtr handle,
            float originX,
            float originZ,
            float cellSize,
            UIntPtr cols,
            UIntPtr rows,
            [In] float[] heightsPtr,
            float friction,
            float restitution);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_clear_heightfield(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void arsa_world_set_query_callbacks(
            IntPtr handle,
            IntPtr userData,
            ArsaGroundHeightFn groundHeightFn,
            ArsaRaycastWorldFn raycastFn);

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

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool arsa_world_compute_host_joint_torques(
            IntPtr handle,
            UIntPtr ragdollId,
            uint coordSpace,
            [Out] float[] outTorques48);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool arsa_world_get_retargeted_bones(
            IntPtr handle,
            UIntPtr ragdollId,
            uint preset,
            [Out] ArsaBoneTransform[] outBones17);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool arsa_world_set_retargeted_target_pose(
            IntPtr handle,
            UIntPtr ragdollId,
            uint preset,
            [In] float[] targetWorldQuats17x4,
            float blendWeight);
    }
}
