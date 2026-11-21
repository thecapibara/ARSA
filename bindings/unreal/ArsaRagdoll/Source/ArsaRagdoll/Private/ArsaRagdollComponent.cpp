#include "ArsaRagdollComponent.h"
#include "GameFramework/Actor.h"
#include "arsa.h"

UArsaRagdollComponent::UArsaRagdollComponent()
{
    PrimaryComponentTick.bCanEverTick = true;
    BoneTransforms.SetNum(ARSA_BONE_COUNT);
    TransformScratch.SetNumZeroed(ARSA_BONE_COUNT * ARSA_TRANSFORM_FLOATS_PER_BODY);
    TelemetryScratch.SetNumZeroed(ARSA_TELEMETRY_FLOATS_PER_RAGDOLL);
}

void UArsaRagdollComponent::BeginPlay()
{
    Super::BeginPlay();

    WorldHandle = arsa_world_create();
    if (!WorldHandle)
    {
        UE_LOG(LogTemp, Error, TEXT("[ARSA] Failed to create ArsaWorldHandle."));
        return;
    }

    arsa_world_spawn_ragdoll_profile(
        WorldHandle,
        0.0f,
        0.0f,
        0.0f,
        HeightMeters,
        MassKg,
        MuscleRatio);
    arsa_world_set_active(WorldHandle, bActiveBalance);

    SyncOutputs();
}

void UArsaRagdollComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    if (WorldHandle)
    {
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

    arsa_world_step(WorldHandle, DeltaTime);
    SyncOutputs();
}

#if WITH_EDITOR
void UArsaRagdollComponent::PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent)
{
    Super::PostEditChangeProperty(PropertyChangedEvent);

    HeightMeters = FMath::Clamp(HeightMeters, 1.40f, 2.40f);
    MassKg = FMath::Clamp(MassKg, 40.0f, 180.0f);
    MuscleRatio = FMath::Clamp(MuscleRatio, 0.25f, 2.0f);

    if (WorldHandle)
    {
        arsa_world_update_all_profiles(WorldHandle, HeightMeters, MassKg, MuscleRatio);
        arsa_world_set_active(WorldHandle, bActiveBalance);
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

int32 UArsaRagdollComponent::FireWeapon(
    FVector OriginUnreal,
    FVector DirectionUnreal,
    EArsaWeaponKind Weapon)
{
    if (!WorldHandle)
    {
        return 0;
    }

    const FTransform OwnerTransform = GetOwner() ? GetOwner()->GetActorTransform() : FTransform::Identity;
    const FVector LocalOrigin = OwnerTransform.InverseTransformPosition(OriginUnreal);
    const FVector LocalDir = OwnerTransform.InverseTransformVectorNoScale(DirectionUnreal);

    float Ox = 0.0f, Oy = 0.0f, Oz = 0.0f;
    float Dx = 0.0f, Dy = 0.0f, Dz = 0.0f;
    UnrealToArsaLocation(LocalOrigin, Ox, Oy, Oz);
    UnrealToArsaDirection(LocalDir, Dx, Dy, Dz);

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

    const FTransform OwnerTransform = GetOwner() ? GetOwner()->GetActorTransform() : FTransform::Identity;
    const FVector LocalOrigin = OwnerTransform.InverseTransformPosition(OriginUnreal);
    const FVector LocalDir = OwnerTransform.InverseTransformVectorNoScale(DirectionUnreal);

    float Ox = 0.0f, Oy = 0.0f, Oz = 0.0f;
    float Dx = 0.0f, Dy = 0.0f, Dz = 0.0f;
    UnrealToArsaLocation(LocalOrigin, Ox, Oy, Oz);
    UnrealToArsaDirection(LocalDir, Dx, Dy, Dz);

    return arsa_world_apply_shove(WorldHandle, Ox, Oy, Oz, Dx, Dy, Dz, Strength);
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

void UArsaRagdollComponent::SyncOutputs()
{
    if (!WorldHandle)
    {
        return;
    }

    const size_t XformFloats = arsa_world_get_transforms(
        WorldHandle,
        TransformScratch.GetData(),
        static_cast<size_t>(TransformScratch.Num()));
    const int32 BodyCount = FMath::Min(
        static_cast<int32>(XformFloats / ARSA_TRANSFORM_FLOATS_PER_BODY),
        BoneTransforms.Num());

    for (int32 i = 0; i < BodyCount; ++i)
    {
        const int32 Base = i * ARSA_TRANSFORM_FLOATS_PER_BODY;
        const FVector Loc = ArsaToUnrealLocation(
            TransformScratch[Base + 0],
            TransformScratch[Base + 1],
            TransformScratch[Base + 2]);
        const FQuat Rot = ArsaToUnrealQuat(
            TransformScratch[Base + 3],
            TransformScratch[Base + 4],
            TransformScratch[Base + 5],
            TransformScratch[Base + 6]);
        BoneTransforms[i] = FTransform(Rot, Loc, FVector::OneVector);
    }

    const size_t TelFloats = arsa_world_get_ragdoll_telemetry(
        WorldHandle,
        TelemetryScratch.GetData(),
        static_cast<size_t>(TelemetryScratch.Num()));
    if (TelFloats >= ARSA_TELEMETRY_FLOATS_PER_RAGDOLL)
    {
        const int32 StateInt = FMath::RoundToInt(TelemetryScratch[7]);
        CachedBehaviorState = static_cast<EArsaBehaviorState>(FMath::Clamp(StateInt, 0, 5));
        CachedHealth = TelemetryScratch[11];
        const int32 WoundInt = FMath::RoundToInt(TelemetryScratch[12]);
        CachedWoundRegion = static_cast<EArsaWoundRegion>(FMath::Clamp(WoundInt, 0, 6));
    }
}
