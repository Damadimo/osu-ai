// Computes osu!standard difficulty attributes (no mods) with the DLLs of the installed osu!lazer client,
// so star ratings match what the client shows. Prints one JSON line per .osu file.
//
//   dotnet run --project tools/lazer_sr -c Release -- data/maps/75.osu other.osu
//
// Needs the .NET 10 SDK (brew "dotnet") and /Applications/osu!.app. After a client update, rebuild.
using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using System.Text.Json;

static class Program
{
    const string AppDir = "/Applications/osu!.app/Contents/MacOS/";

    static int Main(string[] args)
    {
        AssemblyLoadContext.Default.Resolving += (ctx, name) =>
        {
            string p = Path.Combine(AppDir, name.Name + ".dll");
            return File.Exists(p) ? ctx.LoadFromAssemblyPath(p) : null;
        };
        return Run(args);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static int Run(string[] args)
    {
        osu.Framework.Logging.Logger.Enabled = false;
        var ruleset = new osu.Game.Rulesets.Osu.OsuRuleset();
        var gameAsm = typeof(osu.Game.Beatmaps.WorkingBeatmap).Assembly;
        Console.Error.WriteLine($"osu.Game {gameAsm.GetName().Version} ({gameAsm.Location}); runtime {Environment.Version}");
        bool printedVersion = false;
        foreach (var path in args)
        {
            try
            {
                var wb = new osu.Game.Beatmaps.FlatWorkingBeatmap(path);
                var calc = ruleset.CreateDifficultyCalculator(wb);
                if (!printedVersion)
                {
                    var vp = calc.GetType().GetProperty("Version");
                    Console.Error.WriteLine($"{calc.GetType().FullName} Version = {vp?.GetValue(calc)}");
                    printedVersion = true;
                }
                var attr = calc.Calculate();
                var d = new Dictionary<string, object> { ["file"] = Path.GetFileName(path) };
                foreach (var pi in attr.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance))
                {
                    var t = pi.PropertyType;
                    if (t == typeof(double) || t == typeof(int) || t == typeof(float))
                        d[pi.Name] = pi.GetValue(attr);
                }
                foreach (var fi in attr.GetType().GetFields(BindingFlags.Public | BindingFlags.Instance))
                {
                    var t = fi.FieldType;
                    if (t == typeof(double) || t == typeof(int) || t == typeof(float))
                        d[fi.Name] = fi.GetValue(attr);
                }
                var b = wb.Beatmap;
                d["HitObjects"] = b.HitObjects.Count;
                Console.WriteLine(JsonSerializer.Serialize(d));
            }
            catch (Exception e)
            {
                Console.WriteLine(JsonSerializer.Serialize(new Dictionary<string, object> { ["file"] = Path.GetFileName(path), ["error"] = e.ToString() }));
            }
        }
        return 0;
    }
}
