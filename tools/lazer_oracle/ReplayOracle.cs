// `oracle replay <map.osu> <frames.json> ...`: plays replay frames through lazer's real gameplay code (ReplayPlayer
// in a headless host, no window, no input, no network) and prints every judgement as JSON, one line per replay.
// frames.json: {"frames": [[time, x, y, keys], ...]} with keys a bitmask (1 = left button, 2 = right button).
// Besides the replay frames lazer also judges a frame wherever its (headless, 33 ms per step) clock lands between two of
// them; "extra_frames" lists those times so a test can judge exactly the frames lazer did.
// The host's storage (a throwaway realm) lives in a temporary directory that is deleted afterwards.
using System.Runtime.InteropServices;
using System.Runtime.Loader;
using System.Text.Json;
using osu.Framework;
using osu.Framework.Graphics;
using osu.Framework.Graphics.Containers;
using osu.Framework.Platform;
using osu.Framework.Screens;
using osu.Framework.Testing;
using osu.Game;
using osu.Game.Beatmaps;
using osu.Game.IO;
using osu.Game.Online.API;
using osu.Game.Online.API.Requests.Responses;
using osu.Game.Rulesets.Mods;
using osu.Game.Rulesets.Objects;
using osu.Game.Rulesets.Osu;
using osu.Game.Rulesets.Osu.Replays;
using osu.Game.Rulesets.UI;
using osu.Game.Scoring;
using osu.Game.Screens;
using osu.Game.Screens.Play;
using osu.Game.Tests.Visual;
using osuTK;

static class ReplayOracle
{
    public static int Run(string[] args)
    {
        AssemblyLoadContext.Default.ResolvingUnmanagedDll += (_, name) =>
        {
            foreach (string file in new[] { $"lib{name}.dylib", $"{name}.dylib", name })
            {
                string path = Path.Combine(Program.OsuDir, file);
                if (File.Exists(path))
                    return NativeLibrary.Load(path);
            }
            return IntPtr.Zero;
        };

        var jobs = new Queue<(string map, string frames)>();
        for (int i = 0; i + 1 < args.Length; i += 2)
            jobs.Enqueue((Path.GetFullPath(args[i]), Path.GetFullPath(args[i + 1])));

        string work = Path.Combine(Path.GetTempPath(), $"osu-ai-oracle-{Environment.ProcessId}");
        Directory.CreateDirectory(work);
        Directory.SetCurrentDirectory(work);
        try
        {
            using var host = new HeadlessGameHost("oracle", new HostOptions(), realtime: false);
            host.Run(new OracleGame(jobs));
        }
        finally
        {
            Directory.SetCurrentDirectory(Path.GetTempPath());
            try { Directory.Delete(work, true); } catch { /* best effort */ }
        }
        return 0;
    }
}

partial class OracleGame : OsuGameBase
{
    private readonly Queue<(string map, string frames)> jobs;
    private OsuScreenStack stack = null!;
    private ReplayPlayer? player;
    private string currentMap = "";
    private readonly List<object[]> log = new List<object[]>();
    private readonly List<double> frameTimes = new List<double>();
    private readonly HashSet<double> replayTimes = new HashSet<double>();
    private readonly Dictionary<HitObject, int> owner = new Dictionary<HitObject, int>();

    public OracleGame(Queue<(string map, string frames)> jobs)
    {
        this.jobs = jobs;
        API = new DummyAPIAccess();
    }

    protected override void InitialiseFonts()
    {
    }

    protected override void LoadComplete()
    {
        base.LoadComplete();
        Add(stack = new OsuScreenStack { RelativeSizeAxes = Axes.Both });
        Schedule(next);
    }

    private void next()
    {
        if (jobs.Count == 0)
        {
            Exit();
            return;
        }

        var (map, framesFile) = jobs.Dequeue();
        currentMap = map;
        IBeatmap beatmap;
        using (var stream = File.OpenRead(map))
        using (var reader = new LineBufferedReader(stream))
            beatmap = osu.Game.Beatmaps.Formats.Decoder.GetDecoder<Beatmap>(reader).Decode(reader);

        Ruleset.Value = beatmap.BeatmapInfo.Ruleset;
        // A virtual track driven by the host clock: in a non-realtime headless host the game runs as fast as it can.
        Beatmap.Value = new OsuTestScene.ClockBackedTestWorkingBeatmap(beatmap, null, Clock, Audio);
        SelectedMods.Value = Array.Empty<Mod>();

        var replay = new osu.Game.Replays.Replay();
        replayTimes.Clear();
        using (var doc = JsonDocument.Parse(File.ReadAllText(framesFile)))
        {
            foreach (var f in doc.RootElement.GetProperty("frames").EnumerateArray())
            {
                int keys = f[3].GetInt32();
                var actions = new List<OsuAction>();
                if ((keys & 1) != 0) actions.Add(OsuAction.LeftButton);
                if ((keys & 2) != 0) actions.Add(OsuAction.RightButton);
                replay.Frames.Add(new OsuReplayFrame(f[0].GetDouble(), new Vector2(f[1].GetSingle(), f[2].GetSingle()), actions.ToArray()));
                replayTimes.Add(f[0].GetDouble());
            }
        }

        var score = new Score
        {
            Replay = replay,
            ScoreInfo = new ScoreInfo
            {
                User = new APIUser { Username = "oracle" },
                Ruleset = beatmap.BeatmapInfo.Ruleset,
                BeatmapInfo = beatmap.BeatmapInfo,
            },
        };

        log.Clear();
        owner.Clear();
        frameTimes.Clear();
        player = new ReplayPlayer(score, new PlayerConfiguration { AutomaticallySkipIntro = false });
        player.OnLoadComplete += _ =>
        {
            var state = player.GameplayState;
            int index = 0;
            foreach (var h in state.Beatmap.HitObjects)
                own(h, index++);

            var sp = state.ScoreProcessor;
            var hp = state.HealthProcessor;
            player.ChildrenOfType<DrawableRuleset>().First().FrameStableComponents.Add(new FrameLog(frameTimes));
            sp.NewJudgement += r => log.Add(new object[]
            {
                r.TimeAbsolute, owner.GetValueOrDefault(r.HitObject, -1), r.HitObject.GetType().Name, r.Type.ToString(), sp.Combo.Value, hp.Health.Value,
            });
        };
        stack.Push(player);
    }

    private void own(HitObject h, int index)
    {
        owner[h] = index;
        foreach (var n in h.NestedHitObjects)
            own(n, index);
    }

    private int stalled;
    private double lastTime;

    protected override void Update()
    {
        base.Update();

        if (player == null || !player.IsLoaded)
            return;

        // A replay player that fails after its last frame stops its clock and waits for the user: report and move on.
        double now = player.ChildrenOfType<GameplayClockContainer>().First().CurrentTime;
        stalled = now == lastTime ? stalled + 1 : 0;
        lastTime = now;
        if (Environment.GetEnvironmentVariable("ORACLE_DEBUG") != null && stalled == 0)
            Console.Error.WriteLine($"gameplay {now:F0} judged {log.Count}");
        if (!player.GameplayState.HasCompleted && stalled < 5000)
            return;

        var sp = player.GameplayState.ScoreProcessor;
        Console.WriteLine(JsonSerializer.Serialize(new
        {
            file = currentMap,
            accuracy = sp.Accuracy.Value,
            max_combo = sp.HighestCombo.Value,
            total_score = sp.TotalScore.Value,
            rank = sp.Rank.Value.ToString(),
            health = player.GameplayState.HealthProcessor.Health.Value,
            completed = player.GameplayState.HasCompleted,
            judgements = log,
            extra_frames = frameTimes.Where(t => !replayTimes.Contains(t)).Distinct().OrderBy(t => t).ToList(),
        }));
        Console.Out.Flush();

        var done = player;
        player = null;
        done.Exit();
        Schedule(next);
    }
}

/// <summary>Records the gameplay time of every frame lazer updates gameplay at (it runs before the playfield).</summary>
partial class FrameLog : Component
{
    private readonly List<double> times;

    public FrameLog(List<double> times) => this.times = times;

    protected override void Update()
    {
        base.Update();
        times.Add(Time.Current);
    }
}
