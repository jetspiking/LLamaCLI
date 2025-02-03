using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;

namespace LLamaCLI
{
    internal class Program
    {
        static async Task Main(String[] args)
        {
            if (args.Length == 0 || Array.Exists(args, arg => arg.Equals("--help", StringComparison.OrdinalIgnoreCase)))
            {
                CommandLineOptions.PrintHelp();
                return;
            }

            CommandLineOptions options = CommandLineOptions.Parse(args);

            if (String.IsNullOrWhiteSpace(options.ModelFile) ||
                String.IsNullOrWhiteSpace(options.OutputFile) ||
                String.IsNullOrWhiteSpace(options.InputText))
            {
                CommandLineOptions.PrintHelp();
                return;
            }

            SetupNativeLibraryConfig(options);
            await ProcessLLM(options);
        }

        private static void SetupNativeLibraryConfig(CommandLineOptions options)
        {
            if (options.Cuda.Provided)
                NativeLibraryConfig.All.WithCuda(options.Cuda.Value);
            if (options.Vulkan.Provided)
                NativeLibraryConfig.All.WithVulkan(options.Vulkan.Value);
            if (options.AutoFallback.Provided)
                NativeLibraryConfig.All.WithAutoFallback(options.AutoFallback.Value);
            if (options.SkipCheck.Provided)
                NativeLibraryConfig.All.WithAutoFallback(options.SkipCheck.Value);
            if (options.Avx.Provided)
                NativeLibraryConfig.All.WithAvx(Enum.Parse<AvxLevel>(options.Avx.Value));
        }

        private static async Task ProcessLLM(CommandLineOptions options)
        {
            ModelParams modelParams = new ModelParams(options.ModelFile);

            if (options.ContextSize.Provided)
                modelParams.ContextSize = (UInt32)options.ContextSize.Value;

            if (options.MainGpu.Provided)
                modelParams.MainGpu = options.MainGpu.Value;

            if (options.GpuLayerCount.Provided)
                modelParams.GpuLayerCount = options.GpuLayerCount.Value;

            if (options.Threads.Provided)
                modelParams.Threads = options.Threads.Value;

            if (options.BatchThreads.Provided)
                modelParams.BatchThreads = options.BatchThreads.Value;

            if (options.BatchSize.Provided)
                modelParams.BatchSize = (UInt32)options.BatchSize.Value;

            if (options.UBatchSize.Provided)
                modelParams.UBatchSize = (UInt32)options.UBatchSize.Value;

            if (options.UseMemorymap.Provided)
                modelParams.UseMemorymap = options.UseMemorymap.Value;

            if (options.UseMemoryLock.Provided)
                modelParams.UseMemoryLock = options.UseMemoryLock.Value;

            if (options.SeqMax.Provided)
                modelParams.SeqMax = (UInt32)options.SeqMax.Value;

            if (options.SplitMode.Provided)
                modelParams.SplitMode = Enum.Parse<GPUSplitMode>(options.SplitMode.Value);

            if (options.RopeFrequencyBase.Provided)
                modelParams.RopeFrequencyBase = options.RopeFrequencyBase.Value;

            if (options.RopeFrequencyScale.Provided)
                modelParams.RopeFrequencyScale = options.RopeFrequencyScale.Value;

            if (options.FlashAttention.Provided)
                modelParams.FlashAttention = options.FlashAttention.Value;

            using LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            using LLamaContext context = model.CreateContext(modelParams);
            InteractiveExecutor executor = new InteractiveExecutor(context);

            ChatSession session = new(executor);

            InferenceParams inferenceParams = new InferenceParams();

            if (options.MaxTokens.Provided)
                inferenceParams.MaxTokens = options.MaxTokens.Value;

            if (options.TokensKeep.Provided)
                inferenceParams.TokensKeep = options.TokensKeep.Value;

            if (options.AntiPrompts.Provided)
                inferenceParams.AntiPrompts = options.AntiPrompts.Value.Split(',');

            inferenceParams.SamplingPipeline = new DefaultSamplingPipeline();

            IAsyncEnumerable<String> data = session.ChatAsync(new ChatHistory.Message(AuthorRole.User, options.InputText), inferenceParams);

            String output = String.Empty;
            await foreach (String text in data)
            {
                Console.Write(text);
                output += text;
            }

            File.WriteAllText(options.OutputFile, output);
        }


        internal class CommandLineOptions
        {
            public String ModelFile { get; set; } = String.Empty;
            public String OutputFile { get; set; } = String.Empty;
            public String InputText { get; set; } = String.Empty;

            public OptionalArgument<Boolean> Cuda { get; set; } = new OptionalArgument<Boolean>();
            public OptionalArgument<Boolean> Vulkan { get; set; } = new OptionalArgument<Boolean>();
            public OptionalArgument<Boolean> AutoFallback { get; set; } = new OptionalArgument<Boolean>();
            public OptionalArgument<Boolean> SkipCheck { get; set; } = new OptionalArgument<Boolean>();
            public OptionalArgument<String> Avx { get; set; } = new OptionalArgument<String>();
            public OptionalArgument<Int32> ContextSize { get; set; } = new OptionalArgument<Int32>();
            public OptionalArgument<Int32> MainGpu { get; set; } = new OptionalArgument<Int32>();
            public OptionalArgument<Int32> GpuLayerCount { get; set; } = new OptionalArgument<Int32>();
            public OptionalArgument<Int32> Threads { get; set; } = new OptionalArgument<Int32>();
            public OptionalArgument<Int32> BatchThreads { get; set; } = new OptionalArgument<Int32>();
            public OptionalArgument<Int32> BatchSize { get; set; } = new OptionalArgument<Int32>();
            public OptionalArgument<Int32> UBatchSize { get; set; } = new OptionalArgument<Int32>();
            public OptionalArgument<Boolean> UseMemorymap { get; set; } = new OptionalArgument<Boolean>();
            public OptionalArgument<Boolean> UseMemoryLock { get; set; } = new OptionalArgument<Boolean>();

            public OptionalArgument<Int32> MaxTokens { get; set; } = new OptionalArgument<Int32>();
            public OptionalArgument<Int32> TokensKeep { get; set; } = new OptionalArgument<Int32>();
            public OptionalArgument<String> AntiPrompts { get; set; } = new OptionalArgument<String>();

            public OptionalArgument<String> SplitMode { get; set; } = new OptionalArgument<String>();
            public OptionalArgument<Int32> SeqMax { get; set; } = new OptionalArgument<Int32>();
            public OptionalArgument<Single> RopeFrequencyBase { get; set; } = new OptionalArgument<Single>();
            public OptionalArgument<Single> RopeFrequencyScale { get; set; } = new OptionalArgument<Single>();
            public OptionalArgument<Boolean> FlashAttention { get; set; } = new OptionalArgument<Boolean>();

            public static CommandLineOptions Parse(String[] args)
            {
                CommandLineOptions options = new CommandLineOptions();

                for (Int32 i = 0; i < args.Length; i++)
                {
                    String arg = args[i];

                    if (arg.Equals("--help", StringComparison.OrdinalIgnoreCase))
                    {
                        PrintHelp();
                        Environment.Exit(0);
                    }
                    else if (arg.Equals("--modelFile", StringComparison.OrdinalIgnoreCase))
                        options.ModelFile = args[++i];
                    else if (arg.Equals("--outputFile", StringComparison.OrdinalIgnoreCase))
                        options.OutputFile = args[++i];
                    else if (arg.Equals("--inputText", StringComparison.OrdinalIgnoreCase))
                        options.InputText = args[++i];

                    // GPU & Backend Options
                    else if (arg.Equals("--cuda", StringComparison.OrdinalIgnoreCase))
                        options.Cuda = new OptionalArgument<Boolean>(true);
                    else if (arg.Equals("--vulkan", StringComparison.OrdinalIgnoreCase))
                        options.Vulkan = new OptionalArgument<Boolean>(true);
                    else if (arg.Equals("--autoFallback", StringComparison.OrdinalIgnoreCase))
                        options.AutoFallback = new OptionalArgument<Boolean>(true);
                    else if (arg.Equals("--skipCheck", StringComparison.OrdinalIgnoreCase))
                        options.SkipCheck = new OptionalArgument<Boolean>(true);
                    else if (arg.Equals("--avx", StringComparison.OrdinalIgnoreCase))
                        options.Avx = new OptionalArgument<String>(args[++i]);

                    // Model Parameters
                    else if (arg.Equals("--contextSize", StringComparison.OrdinalIgnoreCase))
                        options.ContextSize = new OptionalArgument<Int32>(Int32.Parse(args[++i]));
                    else if (arg.Equals("--mainGpu", StringComparison.OrdinalIgnoreCase))
                        options.MainGpu = new OptionalArgument<Int32>(Int32.Parse(args[++i]));
                    else if (arg.Equals("--gpuLayerCount", StringComparison.OrdinalIgnoreCase))
                        options.GpuLayerCount = new OptionalArgument<Int32>(Int32.Parse(args[++i]));
                    else if (arg.Equals("--threads", StringComparison.OrdinalIgnoreCase))
                        options.Threads = new OptionalArgument<Int32>(Int32.Parse(args[++i]));
                    else if (arg.Equals("--batchThreads", StringComparison.OrdinalIgnoreCase))
                        options.BatchThreads = new OptionalArgument<Int32>(Int32.Parse(args[++i]));
                    else if (arg.Equals("--batchSize", StringComparison.OrdinalIgnoreCase))
                        options.BatchSize = new OptionalArgument<Int32>(Int32.Parse(args[++i]));
                    else if (arg.Equals("--uBatchSize", StringComparison.OrdinalIgnoreCase))
                        options.UBatchSize = new OptionalArgument<Int32>(Int32.Parse(args[++i]));
                    else if (arg.Equals("--useMemorymap", StringComparison.OrdinalIgnoreCase))
                        options.UseMemorymap = new OptionalArgument<Boolean>(true);
                    else if (arg.Equals("--useMemoryLock", StringComparison.OrdinalIgnoreCase))
                        options.UseMemoryLock = new OptionalArgument<Boolean>(true);

                    // Inference Parameters
                    else if (arg.Equals("--maxTokens", StringComparison.OrdinalIgnoreCase))
                        options.MaxTokens = new OptionalArgument<Int32>(Int32.Parse(args[++i]));
                    else if (arg.Equals("--tokensKeep", StringComparison.OrdinalIgnoreCase))
                        options.TokensKeep = new OptionalArgument<Int32>(Int32.Parse(args[++i]));
                    else if (arg.Equals("--antiPrompts", StringComparison.OrdinalIgnoreCase))
                        options.AntiPrompts = new OptionalArgument<String>(args[++i]);

                    // Advanced Model Parameters
                    else if (arg.Equals("--splitMode", StringComparison.OrdinalIgnoreCase))
                        options.SplitMode = new OptionalArgument<String>(args[++i]);
                    else if (arg.Equals("--seqMax", StringComparison.OrdinalIgnoreCase))
                        options.SeqMax = new OptionalArgument<Int32>(Int32.Parse(args[++i]));
                    else if (arg.Equals("--ropeFrequencyBase", StringComparison.OrdinalIgnoreCase))
                        options.RopeFrequencyBase = new OptionalArgument<Single>(Single.Parse(args[++i]));
                    else if (arg.Equals("--ropeFrequencyScale", StringComparison.OrdinalIgnoreCase))
                        options.RopeFrequencyScale = new OptionalArgument<Single>(Single.Parse(args[++i]));
                    else if (arg.Equals("--flashAttention", StringComparison.OrdinalIgnoreCase))
                        options.FlashAttention = new OptionalArgument<Boolean>(true);

                    else
                    {
                        PrintHelp();
                        Environment.Exit(0);
                    }
                }

                return options;
            }

            public static void PrintHelp()
            {
                Console.WriteLine("Usage: LLamaCLI --modelFile <path> --outputFile <path> --inputText <text> [options]");
                Console.WriteLine();
                Console.WriteLine("Required:");
                Console.WriteLine("  --modelFile                 Path to the LLama model file.");
                Console.WriteLine("  --outputFile                Path to the output file.");
                Console.WriteLine("  --inputText                 Input text for processing.");
                Console.WriteLine();
                Console.WriteLine("GPU & Backend Options:");
                Console.WriteLine("  --cuda                      Enable CUDA backend.");
                Console.WriteLine("  --vulkan                    Enable Vulkan backend.");
                Console.WriteLine("  --autoFallback              Allow auto-fallback.");
                Console.WriteLine("  --skipCheck                 Skip validation check.");
                Console.WriteLine("  --avx <level>               AVX level (None, Avx, Avx2, Avx512).");
                Console.WriteLine();
                Console.WriteLine("Model Parameters:");
                Console.WriteLine("  --contextSize <int>         Set context size.");
                Console.WriteLine("  --mainGpu <int>             Set main GPU ID.");
                Console.WriteLine("  --gpuLayerCount <int>       Set GPU layer count.");
                Console.WriteLine("  --threads <int>             Set number of threads.");
                Console.WriteLine("  --batchThreads <int>        Set batch threads.");
                Console.WriteLine("  --batchSize <int>           Set batch size.");
                Console.WriteLine("  --uBatchSize <int>          Set U-batch size.");
                Console.WriteLine("  --useMemorymap              Enable memory mapping.");
                Console.WriteLine("  --useMemoryLock             Enable memory locking.");
                Console.WriteLine();
                Console.WriteLine("Inference Parameters:");
                Console.WriteLine("  --maxTokens <int>           Maximum tokens to generate.");
                Console.WriteLine("  --tokensKeep <int>          Number of tokens to retain.");
                Console.WriteLine("  --antiPrompts <text>        Comma-separated anti-prompts.");
                Console.WriteLine();
                Console.WriteLine("Advanced Model Parameters:");
                Console.WriteLine("  --splitMode <mode>          GPU split mode.");
                Console.WriteLine("  --seqMax <int>              Set max sequence length.");
                Console.WriteLine("  --ropeFrequencyBase <float> Set Rope frequency base.");
                Console.WriteLine("  --ropeFrequencyScale <float> Set Rope frequency scale.");
                Console.WriteLine("  --flashAttention            Enable FlashAttention.");
                Console.WriteLine();
                Console.WriteLine("General:");
                Console.WriteLine("  --help                      Show this help message.");
                Console.WriteLine("=====================================================================");
                Console.WriteLine("This software is available for free & published open source under the MIT license:");
                Console.WriteLine("https://github.com/jetspiking/LLamaCLI");
                Console.WriteLine("=====================================================================");
                Console.WriteLine();
            }

        }
    }

    internal class OptionalArgument<T>
    {
        public Boolean Provided { get; private set; }
        public T? Value { get; private set; }

        public OptionalArgument() { Provided = false; }
        public OptionalArgument(T value) { Value = value; Provided = true; }
    }
}
