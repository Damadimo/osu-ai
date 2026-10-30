// osu!lazer's client.realm: lists skins and osu! ruleset settings, and can change osu! ruleset
// settings (the ones only stored here, e.g. HitAnimations) with --set Key=Value.
// Run only while osu! is closed. Usage: dotnet run -- <client.realm> [schema | --set Key=Value ...]
using System;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Realms;

static class Program
{
    const string OsuDir = "/Applications/osu!.app/Contents/MacOS";

    static int Main(string[] args)
    {
        AppDomain.CurrentDomain.AssemblyResolve += (_, e) =>
        {
            string p = Path.Combine(OsuDir, new AssemblyName(e.Name).Name + ".dll");
            return File.Exists(p) ? Assembly.LoadFrom(p) : null;
        };
        return Dump(args);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]  // Realm.dll must resolve after the handler above exists
    static int Dump(string[] args)
    {
        NativeLibrary.SetDllImportResolver(typeof(Realm).Assembly, (name, asm, path) =>
            name.Contains("realm-wrappers") ? NativeLibrary.Load(Path.Combine(OsuDir, "librealm-wrappers.dylib")) : IntPtr.Zero);
        string file = args.Length > 0 ? args[0]
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Library/Application Support/osu/client.realm");
        var sets = new System.Collections.Generic.Dictionary<string, string>();
        for (int i = 1; i + 1 < args.Length; i++)
            if (args[i] == "--set" && args[i + 1].Contains('='))
                sets[args[i + 1].Split('=')[0]] = args[i + 1].Split('=', 2)[1];
        var config = new RealmConfiguration(file) { IsReadOnly = sets.Count == 0, IsDynamic = true };
        using var realm = Realm.GetInstance(config);
        if (args.Length > 1 && args[1] == "schema")
        {
            foreach (var cls in realm.Schema)
                Console.WriteLine($"{cls.Name}: {string.Join(", ", System.Linq.Enumerable.Select(cls, p => p.Name))}");
            return 0;
        }
        if (sets.Count > 0)
        {
            realm.Write(() =>
            {
                foreach (var row in realm.DynamicApi.All("RulesetSetting"))
                {
                    var o = (IRealmObjectBase)row;
                    string key = o.DynamicApi.Get<string>("Key");
                    if (o.DynamicApi.Get<string>("RulesetName") == "osu" && o.DynamicApi.Get<int>("Variant") == 0 && sets.ContainsKey(key))
                        o.DynamicApi.Set("Value", sets[key]);
                }
            });
        }
        foreach (var skin in realm.DynamicApi.All("Skin"))
        {
            var o = (IRealmObjectBase)skin;
            Console.WriteLine($"skin\t{o.DynamicApi.Get<Guid>("ID")}\t{o.DynamicApi.Get<string>("Name")}\tdeleted={o.DynamicApi.Get<bool>("DeletePending")}");
        }
        foreach (var row in realm.DynamicApi.All("RulesetSetting"))
        {
            var o = (IRealmObjectBase)row;
            Console.WriteLine($"setting\t{o.DynamicApi.Get<string>("RulesetName")}\t{o.DynamicApi.Get<int>("Variant")}\t{o.DynamicApi.Get<string>("Key")}\t{o.DynamicApi.Get<string>("Value")}");
        }
        return 0;
    }
}
