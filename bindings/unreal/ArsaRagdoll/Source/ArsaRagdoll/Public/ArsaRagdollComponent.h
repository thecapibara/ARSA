#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "ArsaRagdollComponent.generated.h"

struct ArsaWorldHandle;

UENUM(BlueprintType)
enum class EArsaBoneId : uint8
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
    RightFoot = 16
};

UENUM(BlueprintType)
enum class EArsaBehaviorState : uint8
{
    Balanced = 0,
    Stumbling = 1,
    Falling = 2,
    Downed = 3,
    Passive = 4,
    Unconscious = 5,
    Recovering = 6 UMETA(DisplayName = "Recovering")
};

UENUM(BlueprintType)
enum class EArsaWoundRegion : uint8
{
    None = 0,
    LeftArm = 1,
    RightArm = 2,
    Torso = 3,
    Head = 4,
    LeftLeg = 5,
    RightLeg = 6
};

UENUM(BlueprintType)
enum class EArsaWeaponKind : uint8
{
    Pistol = 0,
    Shotgun = 1,
    Rifle = 2
};

/**
 * ActorComponent driving a 17-bone ARSA Active Ragdoll simulation in Unreal Engine 5.
 * Converts between ARSA (Right-Handed Y-up, meters) and Unreal Engine (Left-Handed Z-up, centimeters).
 */
UCLASS(ClassGroup = (Physics), meta = (BlueprintSpawnableComponent))
class ARSARAGDOLL_API UArsaRagdollComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UArsaRagdollComponent();

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ARSA|Profile", meta = (ClampMin = "1.40", ClampMax = "2.40"))
    float HeightMeters = 1.78f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ARSA|Profile", meta = (ClampMin = "40.0", ClampMax = "180.0"))
    float MassKg = 75.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ARSA|Profile", meta = (ClampMin = "0.25", ClampMax = "2.0"))
    float MuscleRatio = 1.0f;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ARSA|Control")
    bool bActiveBalance = true;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "ARSA|Control")
    bool bAutoGetUp = true;

    /** Latest 17 bone transforms in Unreal Engine coordinates (relative to owner actor if spawned at origin). */
    UPROPERTY(BlueprintReadOnly, Category = "ARSA|Output")
    TArray<FTransform> BoneTransforms;

    virtual void BeginPlay() override;
    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
    virtual void TickComponent(
        float DeltaTime,
        ELevelTick TickType,
        FActorComponentTickFunction* ThisTickFunction) override;

#if WITH_EDITOR
    virtual void PostEditChangeProperty(FPropertyChangedEvent& PropertyChangedEvent) override;
#endif

    /** Updates anthropometric body proportions, masses, and joint anchors in-place. */
    UFUNCTION(BlueprintCallable, Category = "ARSA|Profile")
    void UpdateBodyProfile(float NewHeightMeters, float NewMassKg, float NewMuscleRatio);

    /** Enables or disables active balance and PD muscle motors. */
    UFUNCTION(BlueprintCallable, Category = "ARSA|Control")
    void SetActiveBalance(bool bEnabled);

    /** Enables or disables automatic procedural get-up from Downed. */
    UFUNCTION(BlueprintCallable, Category = "ARSA|Control")
    void SetAutoGetUp(bool bEnabled);

    /** Sets earthquake seismic magnitude (0.0 .. 9.0, 0.0 disables). */
    UFUNCTION(BlueprintCallable, Category = "ARSA|Control")
    void SetEarthquake(float Magnitude);

    /** Fires a ballistic weapon ray (in Unreal world space) into the ragdoll. Returns hit count. */
    UFUNCTION(BlueprintCallable, Category = "ARSA|Interaction")
    int32 FireWeapon(FVector OriginUnreal, FVector DirectionUnreal, EArsaWeaponKind Weapon);

    /** Applies a broad two-handed shove impulse along an Unreal world-space ray. Returns hit bone index or -1. */
    UFUNCTION(BlueprintCallable, Category = "ARSA|Interaction")
    int32 ApplyShove(FVector OriginUnreal, FVector DirectionUnreal, float Strength = 300.0f);

    /** Applies a sharp localized punch impulse along an Unreal world-space ray. Returns hit bone index or -1. */
    UFUNCTION(BlueprintCallable, Category = "ARSA|Interaction")
    int32 ApplyPunch(FVector Origin, FVector Direction, float Strength = 150.0f);

    /** Returns the current procedural behavioral balance state. */
    UFUNCTION(BlueprintCallable, BlueprintPure, Category = "ARSA|Telemetry")
    EArsaBehaviorState GetBehaviorState() const;

    /** Returns remaining ragdoll health (0.0 .. 100.0). */
    UFUNCTION(BlueprintCallable, BlueprintPure, Category = "ARSA|Telemetry")
    float GetHealth() const;

    /** Returns the most recent anatomical wound region. */
    UFUNCTION(BlueprintCallable, BlueprintPure, Category = "ARSA|Telemetry")
    EArsaWoundRegion GetWoundRegion() const;

    // -------------------------------------------------------------------------
    // Coordinate System Conversions
    // ARSA:   Right-handed, Y-up, meters  (+X left, +Y up, +Z forward)
    // Unreal: Left-handed,  Z-up, cm      (+X forward, +Y right, +Z up)
    // -------------------------------------------------------------------------

    static FORCEINLINE FVector ArsaToUnrealLocation(float X, float Y, float Z)
    {
        return FVector(Z * 100.0f, -X * 100.0f, Y * 100.0f);
    }

    static FORCEINLINE FQuat ArsaToUnrealQuat(float QX, float QY, float QZ, float QW)
    {
        return FQuat(-QZ, QX, -QY, QW);
    }

    static FORCEINLINE void UnrealToArsaLocation(const FVector& LocCm, float& OutX, float& OutY, float& OutZ)
    {
        OutX = -LocCm.Y * 0.01f;
        OutY = LocCm.Z * 0.01f;
        OutZ = LocCm.X * 0.01f;
    }

    static FORCEINLINE void UnrealToArsaDirection(const FVector& Dir, float& OutX, float& OutY, float& OutZ)
    {
        OutX = -Dir.Y;
        OutY = Dir.Z;
        OutZ = Dir.X;
    }

private:
    ArsaWorldHandle* WorldHandle = nullptr;
    TArray<float> TransformScratch;
    TArray<float> TelemetryScratch;

    EArsaBehaviorState CachedBehaviorState = EArsaBehaviorState::Balanced;
    float CachedHealth = 100.0f;
    EArsaWoundRegion CachedWoundRegion = EArsaWoundRegion::None;

    void SyncOutputs();
    void ConvertUnrealRayToArsa(
        const FVector& OriginUnreal,
        const FVector& DirectionUnreal,
        float& OutOx,
        float& OutOy,
        float& OutOz,
        float& OutDx,
        float& OutDy,
        float& OutDz) const;
};
