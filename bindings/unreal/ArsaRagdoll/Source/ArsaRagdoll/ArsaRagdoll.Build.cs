using System;
using System.IO;
using UnrealBuildTool;

public class ArsaRagdoll : ModuleRules
{
    public ArsaRagdoll(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine"
        });

        string PluginRoot = Path.GetFullPath(Path.Combine(ModuleDirectory, "../.."));
        string RepoRoot = Path.GetFullPath(Path.Combine(PluginRoot, "../../.."));

        // Prefer repo bindings/c-abi/include when used in-tree; fall back to ThirdParty/arsa/include.
        string RepoIncludeDir = Path.Combine(RepoRoot, "bindings", "c-abi", "include");
        string ThirdPartyIncludeDir = Path.Combine(PluginRoot, "ThirdParty", "arsa", "include");
        if (Directory.Exists(RepoIncludeDir))
        {
            PublicIncludePaths.Add(RepoIncludeDir);
        }
        if (Directory.Exists(ThirdPartyIncludeDir))
        {
            PublicIncludePaths.Add(ThirdPartyIncludeDir);
        }

        string RepoReleaseDir = Path.Combine(RepoRoot, "target", "release");

        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            string LibDir = ResolveLibDir(PluginRoot, "Win64", RepoReleaseDir, "arsa_c_abi.dll");
            string ImportLib = File.Exists(Path.Combine(LibDir, "arsa_c_abi.dll.lib"))
                ? Path.Combine(LibDir, "arsa_c_abi.dll.lib")
                : Path.Combine(LibDir, "arsa_c_abi.lib");
            string DllPath = Path.Combine(LibDir, "arsa_c_abi.dll");

            PublicAdditionalLibraries.Add(ImportLib);
            RuntimeDependencies.Add("$(BinaryOutputDir)/arsa_c_abi.dll", DllPath);
        }
        else if (Target.Platform == UnrealTargetPlatform.Linux)
        {
            string LibDir = ResolveLibDir(PluginRoot, "Linux", RepoReleaseDir, "libarsa_c_abi.so");
            string SoPath = Path.Combine(LibDir, "libarsa_c_abi.so");

            PublicAdditionalLibraries.Add(SoPath);
            RuntimeDependencies.Add("$(BinaryOutputDir)/libarsa_c_abi.so", SoPath);
        }
        else if (Target.Platform == UnrealTargetPlatform.Mac)
        {
            string LibDir = ResolveLibDir(PluginRoot, "Mac", RepoReleaseDir, "libarsa_c_abi.dylib");
            string DylibPath = Path.Combine(LibDir, "libarsa_c_abi.dylib");

            PublicAdditionalLibraries.Add(DylibPath);
            RuntimeDependencies.Add("$(BinaryOutputDir)/libarsa_c_abi.dylib", DylibPath);
        }
    }

    private static string ResolveLibDir(string PluginRoot, string PlatformName, string RepoReleaseDir, string BinaryName)
    {
        string ThirdPartyLibDir = Path.Combine(PluginRoot, "ThirdParty", "arsa", "lib", PlatformName);
        if (File.Exists(Path.Combine(ThirdPartyLibDir, BinaryName)))
        {
            return ThirdPartyLibDir;
        }
        return RepoReleaseDir;
    }
}
