#include "ArsaRagdollComponent.h"
#include "CollisionQueryParams.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/EngineTypes.h"
#include "Engine/World.h"
#include "GameFramework/Actor.h"
#include "PhysicsEngine/BodyInstance.h"
#include "arsa.h"

namespace
{
    constexpr float ArsaTorqueNmToUnreal = 10000.0f;
    constexpr int32 ArsaJointCount = 16;

    // Parent-to-child bone pairs matching JOINT_BONE_PAIRS in arsa-core (0..16).
    constexpr int32 GArsaJointBonePairs[ArsaJointCount][2] = {
        { ARSA_BONE_PELVIS, ARSA_BONE_LOWER_SPINE },
        { ARSA_BONE_LOWER_SPINE, ARSA_BONE_CHEST },
        { ARSA_BONE_CHEST, ARSA_BONE_NECK },
        { ARSA_BONE_NECK, ARSA_BONE_HEAD },
        { ARSA_BONE_CHEST, ARSA_BONE_LEFT_UPPER_ARM },
        { ARSA_BONE_LEFT_UPPER_ARM, ARSA_BONE_LEFT_FOREARM },
        { ARSA_BONE_LEFT_FOREARM, ARSA_BONE_LEFT_HAND },
        { ARSA_BONE_CHEST, ARSA_BONE_RIGHT_UPPER_ARM },
        { ARSA_BONE_RIGHT_UPPER_ARM, ARSA_BONE_RIGHT_FOREARM },
        { ARSA_BONE_RIGHT_FOREARM, ARSA_BONE_RIGHT_HAND },
        { ARSA_BONE_PELVIS, ARSA_BONE_LEFT_THIGH },
        { ARSA_BONE_LEFT_THIGH, ARSA_BONE_LEFT_SHIN },
        { ARSA_BONE_LEFT_SHIN, ARSA_BONE_LEFT_FOOT },
        { ARSA_BONE_PELVIS, ARSA_BONE_RIGHT_THIGH },
        { ARSA_BONE_RIGHT_THIGH, ARSA_BONE_RIGHT_SHIN },
        { ARSA_BONE_RIGHT_SHIN, ARSA_BONE_RIGHT_FOOT }
    };
}

UArsaRagdollComponent::UArsaRagdollComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    BoneTransforms.SetNum(ARSA_BONE_COUNT);
    RetargetedScratch.SetNumZeroed(ARSA_BONE_COUNT);
    HostTorquesScratch.SetNumZeroed(ArsaJointCount * 3);
    TelemetryScratch.SetNumZeroed(ARSA_TELEMETRY_FLOATS_PER_RAGDOLL);

    BoneNames = {
        FName(TEXT("pelvis")),
        FName(TEXT("spine_01")),
        FName(TEXT("spine_03")),
        FName(TEXT("neck_01")),
        FName(TEXT("head")),
        FName(TEXT("upperarm_l")),
        FName(TEXT("lowerarm_l")),
        FName(TEXT("hand_l")),
        FName(TEXT("upperarm_r")),
        FName(TEXT("lowerarm_r")),
        FName(TEXT("hand_r")),
        FName(TEXT("thigh_l")),
        FName(TEXT("calf_l")),
        FName(TEXT("foot_l")),
        FName(TEXT("thigh_r")),
        FName(TEXT("calf_r")),
        FName(TEXT("foot_r"))
    };
}

void UArsaRagdollComponent::BeginPlay()
{
    Super::BeginPlay();

    if (!TargetSkeletalMesh && GetOwner())
    {
        TargetSkeletalMesh = GetOwner()->FindComponentByClass<USkeletalMeshComponent>();
    }

    WorldHandle = arsa_world_create();
    if (!WorldHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("[ARSA] Failed to create ArsaWorldHandle."));
        return;
    }

    float SpawnX = 0.0f;
    float SpawnY = 0.0f;
    float SpawnZ = 0.0f;
    if (SimulationMode == EArsaSimulationMode::HostPhysics && GetOwner())
    {
        UnrealToArsaLocation(GetOwner()->GetActorLocation(), SpawnX, SpawnY, SpawnZ);
    }

    RagdollId = arsa_world_spawn_ragdoll_profile(
        WorldHandle,
        SpawnX,
        SpawnY,
        SpawnZ,
        HeightMeters,
        MassKg,
        MuscleRatio);
    if (RagdollId == SIZE_MAX)
    {
        UE_LOG(LogTemp, Error, TEXT("[ARSA] Failed to spawn ragdoll profile."));
        arsa_world_destroy(WorldHandle);
        WorldHandle = nullptr;
        return;
    }

    arsa_world_set_active(WorldHandle, bActiveBalance);
    arsa_world_set_auto_get_up(WorldHandle, bAutoGetUp);
    arsa_world_set_intoxication(WorldHandle, FMath::Clamp(Intoxication, 0.0f, 1.0f));
    UpdateChaosQueryCallbacks();

    SyncOutputs();
}

void UArsaRagdollComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (WorldHandle)
    {
        arsa_world_set_query_callbacks(WorldHandle, nullptr, nullptr, nullptr);
        bChaosCallbacksRegistered = false;
        arsa_world_destroy(WorldHandle);
        WorldHandle = nullptr;
    }

    Super::EndPlay(EndPlayReason);
}

void UArsaRagdollComponent::TickComponent(
    float DeltaTime,
    ELevelTick TickType,
    FActorComponentTickFunction* ThisTickFunction)
{
    Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

    if (!WorldHandle)
    {
        return;
    }

    if (SimulationMode == EArsaSimulationMode::HostPhysics)
    {
        StepHostPhysics(DeltaTime);
    }
    else
    {
        arsa_world_step(WorldHandle, DeltaTime);
    }

    SyncOutputs();
}

#if WITH_EDITOR
void UArsaRagdollComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
    Super::PostEditChangeProperty(PropertyChangedEvent);

    HeightMeters = FMath::Clamp(HeightMeters, 1.40f, 2.40f);
    MassKg = FMath::Clamp(MassKg, 40.0f, 180.0f);
    MuscleRatio = FMath::Clamp(MuscleRatio, 0.25f, 2.0f);
    Intoxication = FMath::Clamp(Intoxication, 0.0f, 1.0f);

    if (WorldHandle)
    {
        arsa_world_update_all_profiles(WorldHandle, HeightMeters, MassKg, MuscleRatio);
        arsa_world_set_active(WorldHandle, bActiveBalance);
        arsa_world_set_auto_get_up(WorldHandle, bAutoGetUp);
        arsa_world_set_intoxication(WorldHandle, Intoxication);
        UpdateChaosQueryCallbacks();
    }
}
#endif

void UArsaRagdollComponent::UpdateBodyProfile(
    float NewHeightMeters,
    float NewMassKg,
    float NewMuscleRatio)
{
    HeightMeters = FMath::Clamp(NewHeightMeters, 1.40f, 2.40f);
    MassKg = FMath::Clamp(NewMassKg, 40.0f, 180.0f);
    MuscleRatio = FMath::Clamp(NewMuscleRatio, 0.25f, 2.0f);

    if (WorldHandle)
    {
        arsa_world_update_all_profiles(WorldHandle, HeightMeters, MassKg, MuscleRatio);
    }
}

void UArsaRagdollComponent::SetActiveBalance(bool bEnabled)
{
    bActiveBalance = bEnabled;
    if (WorldHandle)
    {
        arsa_world_set_active(WorldHandle, bActiveBalance);
    }
}

void UArsaRagdollComponent::SetAutoGetUp(bool bEnabled)
{
    bAutoGetUp = bEnabled;
    if (WorldHandle)
    {
        arsa_world_set_auto_get_up(WorldHandle, bAutoGetUp);
    }
}

void UArsaRagdollComponent::SetEarthquake(float Magnitude)
{
    if (WorldHandle)
    {
        arsa_world_set_earthquake(WorldHandle, Magnitude);
    }
}

void UArsaRagdollComponent::SetIntoxication(float Level)
{
    Intoxication = FMath::Clamp(Level, 0.0f, 1.0f);
    if (WorldHandle)
    {
        arsa_world_set_intoxication(WorldHandle, Intoxication);
    }
}

void UArsaRagdollComponent::SetLodFocus(FVector Focus, float ReducedDistMeters, float SleepDistMeters)
{
    if (!WorldHandle)
    {
        return;
    }

    FVector LocalFocus = Focus;
    if (SimulationMode == EArsaSimulationMode::Standalone && GetOwner())
    {
        LocalFocus = GetOwner()->GetActorTransform().InverseTransformPosition(Focus);
    }

    float Ax = 0.0f, Ay = 0.0f, Az = 0.0f;
    UnrealToArsaLocation(LocalFocus, Ax, Ay, Az);
    arsa_world_set_lod_focus(WorldHandle, Ax, Ay, Az, ReducedDistMeters, SleepDistMeters);
}

void UArsaRagdollComponent::ClearLodFocus()
{
    if (WorldHandle)
    {
        arsa_world_clear_lod_focus(WorldHandle);
    }
}

void UArsaRagdollComponent::ConvertUnrealRayToArsa(
    const FVector& OriginUnreal,
    const FVector& DirectionUnreal,
    float& OutOx,
    float& OutOy,
    float& OutOz,
    float& OutDx,
    float& OutDy,
    float& OutDz) const
{
    if (SimulationMode == EArsaSimulationMode::HostPhysics || !GetOwner())
    {
        UnrealToArsaLocation(OriginUnreal, OutOx, OutOy, OutOz);
        UnrealToArsaDirection(DirectionUnreal, OutDx, OutDy, OutDz);
        return;
    }

    const FTransform OwnerTransform = GetOwner()->GetActorTransform();
    const FVector LocalOrigin = OwnerTransform.InverseTransformPosition(OriginUnreal);
    const FVector LocalDir = OwnerTransform.InverseTransformVectorNoScale(DirectionUnreal);

    UnrealToArsaLocation(LocalOrigin, OutOx, OutOy, OutOz);
    UnrealToArsaDirection(LocalDir, OutDx, OutDy, OutDz);
}

int32 UArsaRagdollComponent::FireWeapon(
    FVector OriginUnreal,
    FVector DirectionUnreal,
    EArsaWeaponKind Weapon)
{
    if (!WorldHandle)
    {
        return 0;
    }

    float Ox = 0.0f, Oy = 0.0f, Oz = 0.0f;
    float Dx = 0.0f, Dy = 0.0f, Dz = 0.0f;
    ConvertUnrealRayToArsa(OriginUnreal, DirectionUnreal, Ox, Oy, Oz, Dx, Dy, Dz);

    return arsa_world_fire_weapon(
        WorldHandle,
        Ox,
        Oy,
        Oz,
        Dx,
        Dy,
        Dz,
        static_cast<uint32>(Weapon));
}

int32 UArsaRagdollComponent::ApplyShove(
    FVector OriginUnreal,
    FVector DirectionUnreal,
    float Strength)
{
    if (!WorldHandle)
    {
        return -1;
    }

    float Ox = 0.0f, Oy = 0.0f, Oz = 0.0f;
    float Dx = 0.0f, Dy = 0.0f, Dz = 0.0f;
    ConvertUnrealRayToArsa(OriginUnreal, DirectionUnreal, Ox, Oy, Oz, Dx, Dy, Dz);

    return arsa_world_apply_shove(WorldHandle, Ox, Oy, Oz, Dx, Dy, Dz, Strength);
}

int32 UArsaRagdollComponent::ApplyPunch(
    FVector Origin,
    FVector Direction,
    float Strength)
{
    if (!WorldHandle)
    {
        return -1;
    }

    float Ox = 0.0f, Oy = 0.0f, Oz = 0.0f;
    float Dx = 0.0f, Dy = 0.0f, Dz = 0.0f;
    ConvertUnrealRayToArsa(Origin, Direction, Ox, Oy, Oz, Dx, Dy, Dz);

    return arsa_world_apply_punch(WorldHandle, Ox, Oy, Oz, Dx, Dy, Dz, Strength);
}

EArsaBehaviorState UArsaRagdollComponent::GetBehaviorState() const
{
    return CachedBehaviorState;
}

float UArsaRagdollComponent::GetHealth() const
{
    return CachedHealth;
}

EArsaWoundRegion UArsaRagdollComponent::GetWoundRegion() const
{
    return CachedWoundRegion;
}

void UArsaRagdollComponent::StepHostPhysics(float DeltaTime)
{
    if (!WorldHandle)
    {
        return;
    }

    USkeletalMeshComponent* SkelMesh = TargetSkeletalMesh;
    if (!SkelMesh && GetOwner())
    {
        SkelMesh = GetOwner()->FindComponentByClass<USkeletalMeshComponent>();
        TargetSkeletalMesh = SkelMesh;
    }

    if (SkelMesh)
    {
        const int32 SyncCount = FMath::Min<int32>(ARSA_BONE_COUNT, BoneNames.Num());
        for (int32 i = 0; i < SyncCount; ++i)
        {
            const FBodyInstance* BodyInst = SkelMesh->GetBodyInstance(BoneNames[i]);
            if (!BodyInst)
            {
                continue;
            }

            const FTransform WorldXform = BodyInst->GetUnrealWorldTransform();
            const FVector VelCm = BodyInst->GetUnrealWorldVelocity();
            const FVector AngVelRad = BodyInst->GetUnrealWorldAngularVelocityInRadians();

            float Px = 0.0f, Py = 0.0f, Pz = 0.0f;
            float Qx = 0.0f, Qy = 0.0f, Qz = 0.0f, Qw = 1.0f;
            float Vx = 0.0f, Vy = 0.0f, Vz = 0.0f;
            float Wx = 0.0f, Wy = 0.0f, Wz = 0.0f;

            UnrealToArsaLocation(WorldXform.GetLocation(), Px, Py, Pz);
            UnrealToArsaQuat(WorldXform.GetRotation(), Qx, Qy, Qz, Qw);
            UnrealToArsaLocation(VelCm, Vx, Vy, Vz);
            UnrealToArsaAngular(AngVelRad, Wx, Wy, Wz);

            arsa_world_sync_ragdoll_bone_pose(
                WorldHandle,
                RagdollId,
                static_cast<size_t>(i),
                Px, Py, Pz,
                Qx, Qy, Qz, Qw,
                Vx, Vy, Vz,
                Wx, Wy, Wz);
        }
    }

    arsa_world_step_brain_only(WorldHandle, DeltaTime);

    if (!SkelMesh || BoneNames.Num() < ARSA_BONE_COUNT)
    {
        return;
    }

    if (!arsa_world_compute_host_joint_torques(
            WorldHandle,
            RagdollId,
            ARSA_SPACE_UNREAL_LH_Z_UP,
            HostTorquesScratch.GetData()))
    {
        return;
    }

    for (int32 j = 0; j < ArsaJointCount; ++j)
    {
        const int32 ParentIdx = GArsaJointBonePairs[j][0];
        const int32 ChildIdx = GArsaJointBonePairs[j][1];
        const int32 Base = j * 3;

        const FVector TorqueUE(
            HostTorquesScratch[Base + 0],
            HostTorquesScratch[Base + 1],
            HostTorquesScratch[Base + 2]);
        if (TorqueUE.SizeSquared() < 1e-12f)
        {
            continue;
        }

        const FVector ScaledTorque = TorqueUE * ArsaTorqueNmToUnreal;

        if (FBodyInstance* ChildBody = SkelMesh->GetBodyInstance(BoneNames[ChildIdx]))
        {
            if (ChildBody->IsInstanceSimulatingPhysics())
            {
                ChildBody->AddTorqueInRadians(ScaledTorque, false, false);
            }
        }

        if (FBodyInstance* ParentBody = SkelMesh->GetBodyInstance(BoneNames[ParentIdx]))
        {
            if (ParentBody->IsInstanceSimulatingPhysics())
            {
                ParentBody->AddTorqueInRadians(-ScaledTorque, false, false);
            }
        }
    }
}

void UArsaRagdollComponent::SyncOutputs()
{
    if (!WorldHandle)
    {
        return;
    }

    if (arsa_world_get_retargeted_bones(
            WorldHandle,
            RagdollId,
            static_cast<uint32>(RetargetPreset),
            RetargetedScratch.GetData()))
    {
        const int32 Count = FMath::Min<int32>(ARSA_BONE_COUNT, BoneTransforms.Num());
        for (int32 i = 0; i < Count; ++i)
        {
            const ArsaBoneTransform& B = RetargetedScratch[i];
            FVector Loc;
            FQuat Rot;

            switch (RetargetPreset)
            {
                case EArsaRetargetPreset::UnrealManny:
                    Loc = FVector(B.px, B.py, B.pz);
                    Rot = FQuat(B.qx, B.qy, B.qz, B.qw);
                    break;
                case EArsaRetargetPreset::UnityHumanoid:
                    Loc = ArsaToUnrealLocation(-B.px, B.py, B.pz);
                    Rot = ArsaToUnrealQuat(B.qx, -B.qy, -B.qz, B.qw);
                    break;
                default:
                    Loc = ArsaToUnrealLocation(B.px, B.py, B.pz);
                    Rot = ArsaToUnrealQuat(B.qx, B.qy, B.qz, B.qw);
                    break;
            }

            BoneTransforms[i] = FTransform(Rot, Loc, FVector::OneVector);
        }
    }

    if (arsa_world_get_single_ragdoll_telemetry(
            WorldHandle,
            RagdollId,
            TelemetryScratch.GetData()))
    {
        const int32 StateInt = FMath::RoundToInt(TelemetryScratch[7]);
        CachedBehaviorState = static_cast<EArsaBehaviorState>(FMath::Clamp(StateInt, 0, 6));
        CachedHealth = TelemetryScratch[11];
        const int32 WoundInt = FMath::RoundToInt(TelemetryScratch[12]);
        CachedWoundRegion = static_cast<EArsaWoundRegion>(FMath::Clamp(WoundInt, 0, 6));
    }
}

void UArsaRagdollComponent::UpdateChaosQueryCallbacks()
{
    if (!WorldHandle)
    {
        return;
    }

    if (bEnableChaosWorldQueries && !bChaosCallbacksRegistered)
    {
        arsa_world_set_query_callbacks(
            WorldHandle,
            this,
            &UArsaRagdollComponent::ChaosGroundHeightCallback,
            &UArsaRagdollComponent::ChaosRaycastCallback);
        bChaosCallbacksRegistered = true;
    }
    else if (!bEnableChaosWorldQueries && bChaosCallbacksRegistered)
    {
        arsa_world_set_query_callbacks(WorldHandle, nullptr, nullptr, nullptr);
        bChaosCallbacksRegistered = false;
    }
}

bool UArsaRagdollComponent::ChaosGroundHeightCallback(
    void* UserData,
    float X,
    float Z,
    float MaxY,
    float* OutY,
    float OutNormal[3])
{
    UArsaRagdollComponent* Comp = static_cast<UArsaRagdollComponent*>(UserData);
    if (!Comp || !OutY || !OutNormal)
    {
        return false;
    }

    UWorld* World = Comp->GetWorld();
    if (!World)
    {
        return false;
    }

    AActor* Owner = Comp->GetOwner();
    const bool bUseOwnerTransform = (Comp->SimulationMode == EArsaSimulationMode::Standalone && Owner != nullptr);
    const FTransform OwnerTransform = bUseOwnerTransform ? Owner->GetActorTransform() : FTransform::Identity;

    const FVector LocalTop = ArsaToUnrealLocation(X, MaxY, Z);
    const FVector WorldStart = OwnerTransform.TransformPosition(LocalTop);
    const FVector WorldDown = OwnerTransform.TransformVectorNoScale(-FVector::UpVector).GetSafeNormal();
    const FVector WorldEnd = WorldStart + WorldDown * 5000.0f;

    FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(ArsaGroundHeight), false);
    if (Owner)
    {
        QueryParams.AddIgnoredActor(Owner);
    }

    FHitResult Hit;
    if (!World->LineTraceSingleByChannel(Hit, WorldStart, WorldEnd, ECC_WorldStatic, QueryParams))
    {
        return false;
    }

    const FVector HitLocal = OwnerTransform.InverseTransformPosition(Hit.ImpactPoint);
    const FVector NormalLocal = OwnerTransform.InverseTransformVectorNoScale(Hit.ImpactNormal).GetSafeNormal();

    float HitAx = 0.0f, HitAy = 0.0f, HitAz = 0.0f;
    UnrealToArsaLocation(HitLocal, HitAx, HitAy, HitAz);
    *OutY = HitAy;

    UnrealToArsaDirection(NormalLocal, OutNormal[0], OutNormal[1], OutNormal[2]);
    return true;
}

bool UArsaRagdollComponent::ChaosRaycastCallback(
    void* UserData,
    const float Origin[3],
    const float Dir[3],
    float MaxDist,
    float* OutDist,
    float OutNormal[3])
{
    UArsaRagdollComponent* Comp = static_cast<UArsaRagdollComponent*>(UserData);
    if (!Comp || !Origin || !Dir || !OutDist || !OutNormal || MaxDist <= 0.0f)
    {
        return false;
    }

    UWorld* World = Comp->GetWorld();
    if (!World)
    {
        return false;
    }

    AActor* Owner = Comp->GetOwner();
    const bool bUseOwnerTransform = (Comp->SimulationMode == EArsaSimulationMode::Standalone && Owner != nullptr);
    const FTransform OwnerTransform = bUseOwnerTransform ? Owner->GetActorTransform() : FTransform::Identity;

    const FVector LocalOrigin = ArsaToUnrealLocation(Origin[0], Origin[1], Origin[2]);
    const FVector LocalDir = ArsaToUnrealDirection(Dir[0], Dir[1], Dir[2]);

    const FVector WorldStart = OwnerTransform.TransformPosition(LocalOrigin);
    const FVector WorldDir = OwnerTransform.TransformVectorNoScale(LocalDir).GetSafeNormal();
    if (WorldDir.IsNearlyZero())
    {
        return false;
    }

    const FVector WorldEnd = WorldStart + WorldDir * (MaxDist * 100.0f);

    FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(ArsaRaycastWorld), false);
    if (Owner)
    {
        QueryParams.AddIgnoredActor(Owner);
    }

    FHitResult Hit;
    if (!World->LineTraceSingleByChannel(Hit, WorldStart, WorldEnd, ECC_WorldStatic, QueryParams))
    {
        return false;
    }

    const FVector NormalLocal = OwnerTransform.InverseTransformVectorNoScale(Hit.ImpactNormal).GetSafeNormal();
    *OutDist = Hit.Distance * 0.01f;
    UnrealToArsaDirection(NormalLocal, OutNormal[0], OutNormal[1], OutNormal[2]);
    return true;
}
