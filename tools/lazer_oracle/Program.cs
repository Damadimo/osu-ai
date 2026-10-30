// Read-only oracle: loads the installed osu!lazer assemblies and dumps gameplay constants / processed beatmaps as JSON.
using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;

static class Program
{
    internal const string OsuDir = "/Applications/osu!.app/Contents/MacOS";

    static int Main(string[] args)
    {
        AppDomain.CurrentDomain.AssemblyResolve += (_, e) =>
        {
            string name = new AssemblyName(e.Name).Name!;
            string p = Path.Combine(OsuDir, name + ".dll");
            return File.Exists(p) ? Assembly.LoadFrom(p) : null;
        };
        return Oracle.Run(args);
    }
}

static class Oracle
{
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static int Run(string[] args)
    {
        var opts = new JsonSerializerOptions { WriteIndented = false };
        if (args.Length > 2 && args[0] == "replay")
            return ReplayOracle.Run(args.Skip(1).ToArray());

        if (args.Length > 0 && args[0] == "tables")
        {
            var rows = new List<object>();
            for (int k = 0; k <= 100; k++)
            {
                // parse exactly like LegacyBeatmapDecoder (float.Parse of the decimal string)
                float v = float.Parse((k / 10.0).ToString("0.0", CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
                var hw = new osu.Game.Rulesets.Osu.Scoring.OsuHitWindows();
                hw.SetDifficulty(v);
                float scale = osu.Game.Rulesets.Objects.Legacy.LegacyRulesetExtensions.CalculateScaleFromCircleSize(v, true);
                int preempt = osu.Game.Beatmaps.IBeatmapDifficultyInfo.DifficultyRangeInt(v, osu.Game.Rulesets.Osu.Objects.OsuHitObject.PREEMPT_RANGE);
                rows.Add(new
                {
                    value = k / 10.0,
                    great = hw.WindowFor(osu.Game.Rulesets.Scoring.HitResult.Great),
                    ok = hw.WindowFor(osu.Game.Rulesets.Scoring.HitResult.Ok),
                    meh = hw.WindowFor(osu.Game.Rulesets.Scoring.HitResult.Meh),
                    miss = hw.WindowFor(osu.Game.Rulesets.Scoring.HitResult.Miss),
                    scale = (double)scale,
                    radius = (double)(osu.Game.Rulesets.Osu.Objects.OsuHitObject.OBJECT_RADIUS * scale),
                    preempt,
                });
            }
            Console.WriteLine(JsonSerializer.Serialize(rows, opts));
            return 0;
        }

        if (args.Length > 0 && args[0] == "maps")
        {
            osu.Game.Beatmaps.Formats.Decoder.RegisterDependencies(new osu.Game.Rulesets.AssemblyRulesetStore());
            var ruleset = new osu.Game.Rulesets.Osu.OsuRuleset();
            foreach (string file in args.Skip(1))
            {
                try
                {
                    Console.WriteLine(JsonSerializer.Serialize(DumpMap(file, ruleset), opts));
                }
                catch (Exception ex)
                {
                    Console.WriteLine(JsonSerializer.Serialize(new { file, error = ex.ToString() }, opts));
                }
            }
            return 0;
        }

        if (args.Length > 0 && args[0] == "health")
        {
            osu.Game.Beatmaps.Formats.Decoder.RegisterDependencies(new osu.Game.Rulesets.AssemblyRulesetStore());
            var ruleset = new osu.Game.Rulesets.Osu.OsuRuleset();
            foreach (string file in args.Skip(1))
                Console.WriteLine(JsonSerializer.Serialize(ScriptedHealth(file, ruleset), opts));
            return 0;
        }

        if (args.Length > 1 && args[0] == "failtest")
        {
            osu.Game.Beatmaps.Formats.Decoder.RegisterDependencies(new osu.Game.Rulesets.AssemblyRulesetStore());
            var ruleset = new osu.Game.Rulesets.Osu.OsuRuleset();
            var working = new osu.Game.Beatmaps.FlatWorkingBeatmap(args[1]);
            var playable = working.GetPlayableBeatmap(ruleset.RulesetInfo, Array.Empty<osu.Game.Rulesets.Mods.Mod>());
            var slider = playable.HitObjects.OfType<osu.Game.Rulesets.Osu.Objects.Slider>().First();
            var circle = playable.HitObjects.OfType<osu.Game.Rulesets.Osu.Objects.HitCircle>().First();
            var spinner = playable.HitObjects.OfType<osu.Game.Rulesets.Osu.Objects.Spinner>().FirstOrDefault();
            var cases = new List<(string name, double start, osu.Game.Rulesets.Objects.HitObject obj, osu.Game.Rulesets.Scoring.HitResult type)>
            {
                ("health 0 then circle Great", 0, circle, osu.Game.Rulesets.Scoring.HitResult.Great),
                ("health 0 then circle Meh", 0, circle, osu.Game.Rulesets.Scoring.HitResult.Meh),
                ("health 0 then slider tail IgnoreMiss", 0, slider.TailCircle, osu.Game.Rulesets.Scoring.HitResult.IgnoreMiss),
                ("health 0 then slider IgnoreHit", 0, slider, osu.Game.Rulesets.Scoring.HitResult.IgnoreHit),
                ("health 0.05 then circle Miss", 0.05, circle, osu.Game.Rulesets.Scoring.HitResult.Miss),
                ("health 0.2 then circle Miss", 0.2, circle, osu.Game.Rulesets.Scoring.HitResult.Miss),
            };
            if (spinner != null)
                cases.Add(("health 0 then spinner tick SmallBonus", 0, spinner.NestedHitObjects[0], osu.Game.Rulesets.Scoring.HitResult.SmallBonus));
            foreach (var c in cases)
            {
                var hp = new osu.Game.Rulesets.Osu.Scoring.OsuHealthProcessor(playable.HitObjects[0].StartTime);
                hp.ApplyBeatmap(playable);
                hp.Health.Value = c.start;
                var res = new osu.Game.Rulesets.Judgements.JudgementResult(c.obj, c.obj.Judgement) { Type = c.type };
                hp.ApplyResult(res);
                Console.WriteLine($"{c.name}: health after {hp.Health.Value:F4} failed={hp.HasFailed}");
            }
            return 0;
        }

        Console.Error.WriteLine("usage: oracle tables | oracle maps <file.osu>... | oracle health <file.osu>... | oracle failtest <file.osu> | oracle replay <file.osu> <frames.json>...");
        return 2;
    }

    static object Pos(osuTK.Vector2 v) => new[] { (double)v.X, (double)v.Y };

    // Deterministic scripted results fed through the real OsuHealthProcessor in live (non-simulating) mode.
    // Health is reset to 0.5 before each judgement so the per-judgement increase is observable without clamping/drain.
    static object ScriptedHealth(string file, osu.Game.Rulesets.Osu.OsuRuleset ruleset)
    {
        var working = new osu.Game.Beatmaps.FlatWorkingBeatmap(file);
        var playable = working.GetPlayableBeatmap(ruleset.RulesetInfo, Array.Empty<osu.Game.Rulesets.Mods.Mod>());
        var hp = new osu.Game.Rulesets.Osu.Scoring.OsuHealthProcessor(playable.HitObjects[0].StartTime);
        hp.ApplyBeatmap(playable);
        var R = typeof(osu.Game.Rulesets.Scoring.HitResult);
        var basic = new[] { osu.Game.Rulesets.Scoring.HitResult.Great, osu.Game.Rulesets.Scoring.HitResult.Great, osu.Game.Rulesets.Scoring.HitResult.Ok, osu.Game.Rulesets.Scoring.HitResult.Great, osu.Game.Rulesets.Scoring.HitResult.Meh, osu.Game.Rulesets.Scoring.HitResult.Great, osu.Game.Rulesets.Scoring.HitResult.Miss };
        int k = 0;
        var log = new List<object>();
        int idx = 0;
        foreach (var h in playable.HitObjects)
        {
            bool anyHit = false;
            foreach (var n in h.NestedHitObjects)
            {
                var j = n.Judgement;
                var res = new osu.Game.Rulesets.Judgements.JudgementResult(n, j);
                k++;
                if (j.MaxResult == osu.Game.Rulesets.Scoring.HitResult.Great) res.Type = basic[k % basic.Length];
                else res.Type = (k % 5 == 0) ? j.MinResult : j.MaxResult;
                anyHit |= osu.Game.Rulesets.Scoring.HitResultExtensions.IsHit(res.Type);
                hp.Health.Value = 0.5;
                hp.ApplyResult(res);
                log.Add(new object[] { idx, n.GetType().Name, res.Type.ToString(), hp.Health.Value - 0.5 });
            }
            var pj = h.Judgement;
            var pres = new osu.Game.Rulesets.Judgements.JudgementResult(h, pj);
            k++;
            if (h is osu.Game.Rulesets.Osu.Objects.Slider) pres.Type = anyHit ? pj.MaxResult : pj.MinResult;
            else pres.Type = basic[k % basic.Length];
            hp.Health.Value = 0.5;
            hp.ApplyResult(pres);
            log.Add(new object[] { idx, h.GetType().Name, pres.Type.ToString(), hp.Health.Value - 0.5 });
            idx++;
        }
        return new { file, log };
    }

    static object DumpMap(string file, osu.Game.Rulesets.Osu.OsuRuleset ruleset)
    {
        var working = new osu.Game.Beatmaps.FlatWorkingBeatmap(file);
        var playable = working.GetPlayableBeatmap(ruleset.RulesetInfo, Array.Empty<osu.Game.Rulesets.Mods.Mod>());
        var objs = playable.HitObjects.Cast<osu.Game.Rulesets.Osu.Objects.OsuHitObject>().ToList();

        var hp = new osu.Game.Rulesets.Osu.Scoring.OsuHealthProcessor(playable.HitObjects[0].StartTime);
        hp.ApplyBeatmap(playable);

        var sp = new osu.Game.Rulesets.Osu.Scoring.OsuScoreProcessor();
        sp.ApplyBeatmap(playable);

        var outObjs = new List<object>();
        foreach (var h in objs)
        {
            var nested = h.NestedHitObjects.Cast<osu.Game.Rulesets.Osu.Objects.OsuHitObject>()
                          .Select(n => new
                          {
                              type = n.GetType().Name,
                              t = n.StartTime,
                              pos = Pos(n.StackedPosition),
                              max = n.Judgement.MaxResult.ToString(),
                          }).ToList();
            object extra = null;
            switch (h)
            {
                case osu.Game.Rulesets.Osu.Objects.Slider s:
                    extra = new
                    {
                        spans = s.RepeatCount + 1,
                        span_duration = s.SpanDuration,
                        velocity = s.Velocity,
                        tick_distance = s.TickDistance,
                        distance = s.Path.Distance,
                        end_pos = Pos(s.StackedEndPosition),
                    };
                    break;
                case osu.Game.Rulesets.Osu.Objects.Spinner sp2:
                    extra = new { spins_required = sp2.SpinsRequired, max_bonus_spins = sp2.MaximumBonusSpins, spins_for_bonus = sp2.SpinsRequiredForBonus };
                    break;
            }
            outObjs.Add(new
            {
                type = h.GetType().Name,
                t = h.StartTime,
                end = osu.Game.Rulesets.Objects.HitObjectExtensions.GetEndTime(h),
                pos = Pos(h.Position),
                stack = h.StackHeight,
                spos = Pos(h.StackedPosition),
                nc = h.NewCombo,
                last_in_combo = h.LastInCombo,
                preempt = h.TimePreempt,
                radius = h.Radius,
                extra,
                nested,
            });
        }

        return new
        {
            file,
            version = playable.BeatmapVersion,
            hp = playable.Difficulty.DrainRate,
            cs = playable.Difficulty.CircleSize,
            od = playable.Difficulty.OverallDifficulty,
            ar = playable.Difficulty.ApproachRate,
            stack_leniency = playable.StackLeniency,
            drain_rate = hp.DrainRate,
            max_combo = sp.MaximumCombo,
            max_stats = sp.MaximumStatistics.ToDictionary(kv => kv.Key.ToString(), kv => kv.Value),
            breaks = playable.Breaks.Select(b => new[] { b.StartTime, b.EndTime }).ToList(),
            objects = outObjs,
        };
    }
}
