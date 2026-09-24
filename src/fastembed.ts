import { AddedToken, Tokenizer } from "@anush008/tokenizers";
import fs, { PathLike } from "fs";
import * as ort from "onnxruntime-node";
import path from "path";
import Progress from "progress";
import { Readable, Transform } from "stream";
import { pipeline } from "stream/promises";
import { ReadableStream } from "stream/web";
import { downloadFileToCacheDir } from "@huggingface/hub";

export enum ExecutionProvider {
  CPU = "cpu",
  CUDA = "cuda",
  WebGL = "webgl",
  WASM = "wasm",
  XNNPACK = "xnnpack",
}

export enum EmbeddingModel {
  AllMiniLML6V2 = "fast-all-MiniLM-L6-v2",
  BGEBaseEN = "fast-bge-base-en",
  BGEBaseENV15 = "fast-bge-base-en-v1.5",
  BGESmallEN = "fast-bge-small-en",
  BGESmallENV15 = "fast-bge-small-en-v1.5",
  BGESmallZH = "fast-bge-small-zh-v1.5",
  MLE5Large = "fast-multilingual-e5-large",
  CUSTOM = "custom",
}

export enum Pooling {
  CLS = "cls",
  Mean = "mean",
}

export enum SparseEmbeddingModel {
  SpladePPEnV1 = "prithivida/Splade_PP_en_v1",
  CUSTOM = "custom",
}

// Sparse embedding types
export type SparseVector = {
  values: number[],
  indices: number[]
}

export interface InitOptionsBase {
  executionProviders?: ExecutionProvider[];
  maxLength?: number;
  cacheDir?: string;
  showDownloadProgress?: boolean;
}

interface ModelInfo {
  model: EmbeddingModel;
  dim: number;
  description: string;
}

interface ModelSource {
  // Hugging Face repository the model files are downloaded from
  repo: string;
  // Path of the ONNX model file inside the repository
  modelFile: string;
  // Extra files the model needs besides the tokenizer files, e.g. external weights
  additionalFiles?: string[];
  pooling: Pooling;
}

const TOKENIZER_FILES = [
  "tokenizer.json",
  "tokenizer_config.json",
  "config.json",
  "special_tokens_map.json",
];

const MODEL_SOURCES: Record<
  Exclude<EmbeddingModel, EmbeddingModel.CUSTOM>,
  ModelSource
> = {
  [EmbeddingModel.AllMiniLML6V2]: {
    repo: "Qdrant/all-MiniLM-L6-v2-onnx",
    modelFile: "model.onnx",
    pooling: Pooling.Mean,
  },
  [EmbeddingModel.BGEBaseEN]: {
    repo: "Qdrant/fast-bge-base-en",
    modelFile: "model_optimized.onnx",
    pooling: Pooling.CLS,
  },
  [EmbeddingModel.BGEBaseENV15]: {
    repo: "Qdrant/bge-base-en-v1.5-onnx-Q",
    modelFile: "model_optimized.onnx",
    pooling: Pooling.CLS,
  },
  [EmbeddingModel.BGESmallEN]: {
    repo: "Qdrant/bge-small-en",
    modelFile: "model_optimized.onnx",
    pooling: Pooling.CLS,
  },
  [EmbeddingModel.BGESmallENV15]: {
    repo: "Qdrant/bge-small-en-v1.5-onnx-Q",
    modelFile: "model_optimized.onnx",
    pooling: Pooling.CLS,
  },
  [EmbeddingModel.BGESmallZH]: {
    repo: "Qdrant/bge-small-zh-v1.5",
    modelFile: "model_optimized.onnx",
    pooling: Pooling.CLS,
  },
  [EmbeddingModel.MLE5Large]: {
    repo: "Qdrant/multilingual-e5-large-onnx",
    modelFile: "model.onnx",
    additionalFiles: ["model.onnx_data"],
    pooling: Pooling.Mean,
  },
};

const HF_ENDPOINT = process.env.HF_ENDPOINT || "https://huggingface.co";

interface SparseModelInfo {
  model: SparseEmbeddingModel;
  vocabSize: number;
  description: string;
}

type ModelInput = Record<string, ort.Tensor>;

function normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((acc, val) => acc + val * val, 0));
  const epsilon = 1e-12;

  return v.map((val) => val / Math.max(norm, epsilon));
}

// Pools the token embeddings of each text in the batch into a single embedding
function poolEmbeddings(
  data: Float32Array,
  dimensions: [number, number, number],
  attentionMask: bigint[][],
  pooling: Pooling
): number[][] {
  const [batchSize, seqLen, hiddenSize] = dimensions;

  return Array.from({ length: batchSize }, (_, batchIdx) => {
    const offset = batchIdx * seqLen * hiddenSize;
    if (pooling === Pooling.CLS) {
      return Array.from(data.subarray(offset, offset + hiddenSize));
    }

    // Mean of the token embeddings, ignoring padding tokens
    const sum = new Array<number>(hiddenSize).fill(0);
    let tokenCount = 0;
    for (let seqIdx = 0; seqIdx < seqLen; seqIdx++) {
      if (attentionMask[batchIdx][seqIdx] === 0n) continue;
      tokenCount++;
      const tokenOffset = offset + seqIdx * hiddenSize;
      for (let i = 0; i < hiddenSize; i++) {
        sum[i] += data[tokenOffset + i];
      }
    }
    return sum.map((val) => val / Math.max(tokenCount, 1e-9));
  });
}

// Cas standard
export interface InitStandardOptions extends InitOptionsBase {
  model: Exclude<EmbeddingModel, EmbeddingModel.CUSTOM>;
  modelAbsoluteDirPath?: undefined;
  modelName?: string;
  pooling?: undefined;
}

// Cas custom
export interface InitCustomOptions extends InitOptionsBase {
  model: EmbeddingModel.CUSTOM;
  modelAbsoluteDirPath: fs.PathLike;
  modelName: string;
  // How token embeddings are pooled into one embedding. Defaults to CLS
  pooling?: Pooling;
}
export type InitOptions = InitStandardOptions | InitCustomOptions;

// Sparse embedding init options
export interface InitSparseStandardOptions extends InitOptionsBase {
  model: Exclude<SparseEmbeddingModel, SparseEmbeddingModel.CUSTOM>;
  modelAbsoluteDirPath?: undefined;
  modelName?: string;
}

export interface InitSparseCustomOptions extends InitOptionsBase {
  model: SparseEmbeddingModel.CUSTOM;
  modelAbsoluteDirPath: fs.PathLike;
  modelName: string;
}

export type InitSparseOptions =
  | InitSparseStandardOptions
  | InitSparseCustomOptions;

abstract class Embedding {
  abstract listSupportedModels(): ModelInfo[];

  abstract embed(
    texts: string[],
    batchSize?: number
  ): AsyncGenerator<number[][], void, unknown>;

  abstract passageEmbed(
    texts: string[],
    batchSize: number
  ): AsyncGenerator<number[][], void, unknown>;

  abstract queryEmbed(query: string): Promise<number[]>;
}

abstract class SparseEmbedding {
  abstract listSupportedModels(): SparseModelInfo[];

  abstract embed(
    texts: string[],
    batchSize?: number
  ): AsyncGenerator<SparseVector[], void, unknown>;

  abstract passageEmbed(
    texts: string[],
    batchSize: number
  ): AsyncGenerator<SparseVector[], void, unknown>;

  abstract queryEmbed(query: string): Promise<SparseVector>;
}

export class FlagEmbedding extends Embedding {
  private constructor(
    private tokenizer: Tokenizer,
    private session: ort.InferenceSession,
    private model: EmbeddingModel,
    private pooling: Pooling
  ) {
    super();
  }
  static async init(options: InitStandardOptions): Promise<FlagEmbedding>;
  static async init(options: InitCustomOptions): Promise<FlagEmbedding>;
  static async init({
    model = EmbeddingModel.BGESmallENV15,
    executionProviders = [ExecutionProvider.CPU],
    maxLength = 512,
    cacheDir = "local_cache",
    showDownloadProgress = true,
    modelAbsoluteDirPath = "",
    modelName = "",
    pooling = Pooling.CLS,
  }: Partial<InitOptions> = {}) {
    if (model === EmbeddingModel.CUSTOM) {
      if (!modelAbsoluteDirPath) {
        throw new Error(
          "For custom model, modelAbsoluteDirPath is required in FlagEmbedding.init"
        );
      }
      if (!modelName) {
        throw new Error(
          "For custom model, modelName is required in FlagEmbedding.init"
        );
      }
    }
    const modelDir =
      model === EmbeddingModel.CUSTOM
        ? modelAbsoluteDirPath
        : await FlagEmbedding.retrieveModel(
            model,
            cacheDir,
            showDownloadProgress
          );

    const tokenizer = this.loadTokenizer(modelDir, maxLength);
    const defaultModelName =
      model === EmbeddingModel.CUSTOM ? "" : MODEL_SOURCES[model].modelFile;
    const modelPath = path.join(
      modelDir.toString(),
      modelName || defaultModelName
    );
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Model file not found at ${modelPath}`);
    }
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders,
      graphOptimizationLevel: "all",
    });
    return new FlagEmbedding(
      tokenizer,
      session,
      model,
      model === EmbeddingModel.CUSTOM ? pooling : MODEL_SOURCES[model].pooling
    );
  }

  private static loadTokenizer(
    modelDir: fs.PathLike,
    maxLength: number
  ): Tokenizer {
    const tokenizerPath = path.join(modelDir.toString(), "tokenizer.json");
    if (!fs.existsSync(tokenizerPath)) {
      throw new Error(`Tokenizer file not found at ${tokenizerPath}`);
    }

    const configPath = path.join(modelDir.toString(), "config.json");
    if (!fs.existsSync(configPath)) {
      throw new Error(`Config file not found at ${configPath}`);
    }
    const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));

    const tokenizerFilePath = path.join(
      modelDir.toString(),
      "tokenizer_config.json"
    );
    if (!fs.existsSync(tokenizerFilePath)) {
      throw new Error(`Tokenizer file not found at ${tokenizerFilePath}`);
    }
    const tokenizerConfig = JSON.parse(
      fs.readFileSync(tokenizerFilePath, "utf-8")
    );
    maxLength = Math.min(maxLength, tokenizerConfig["model_max_length"]);

    const tokensMapPath = path.join(
      modelDir.toString(),
      "special_tokens_map.json"
    );
    if (!fs.existsSync(tokensMapPath)) {
      throw new Error(`Tokens map file not found at ${tokensMapPath}`);
    }
    const tokensMap = JSON.parse(fs.readFileSync(tokensMapPath, "utf-8"));

    const tokenizer = Tokenizer.fromFile(tokenizerPath);

    tokenizer.setTruncation(maxLength);
    tokenizer.setPadding({
      maxLength,
      padId: config["pad_token_id"],
      padToken: tokenizerConfig["pad_token"],
    });

    for (let token of Object.values(tokensMap)) {
      if (typeof token === "string") {
        tokenizer.addSpecialTokens([token]);
      } else if (isAddedTokenMap(token)) {
        const addedToken = new AddedToken(token["content"], true, {
          singleWord: token["single_word"],
          leftStrip: token["lstrip"],
          rightStrip: token["rstrip"],
          normalized: token["normalized"],
        });
        tokenizer.addAddedTokens([addedToken]);
      }
    }
    return tokenizer;
  }

  private static async downloadFileFromHF(
    outputFilePath: string,
    repo: string,
    fileName: string,
    showDownloadProgress: boolean = true
  ): Promise<void> {
    if (fs.existsSync(outputFilePath)) {
      return;
    }

    const url = `${HF_ENDPOINT}/${repo}/resolve/main/${fileName}`;
    const response = await fetch(url);
    if (!response.ok || !response.body) {
      throw new Error(
        `Failed to download ${url}: ${response.status} ${response.statusText}`
      );
    }

    const totalSizeInBytes = parseInt(
      response.headers.get("content-length") || "0",
      10
    );
    const progressBar =
      showDownloadProgress && totalSizeInBytes > 0
        ? new Progress(`Downloading ${repo}/${fileName} [:bar] :percent :etas`, {
            complete: "=",
            width: 20,
            total: totalSizeInBytes,
          })
        : undefined;

    // Write to a temporary file first, so an interrupted download is never mistaken for a complete one
    const partialFilePath = `${outputFilePath}.part`;
    fs.mkdirSync(path.dirname(outputFilePath), {
      recursive: true,
      mode: 0o777,
    });
    try {
      await pipeline(
        Readable.fromWeb(response.body as ReadableStream<Uint8Array>),
        new Transform({
          transform(chunk, _encoding, callback) {
            progressBar?.tick(chunk.length);
            callback(null, chunk);
          },
        }),
        fs.createWriteStream(partialFilePath)
      );
    } catch (error) {
      fs.rmSync(partialFilePath, { force: true });
      throw error;
    }
    fs.renameSync(partialFilePath, outputFilePath);
  }

  private static async retrieveModel(
    model: Exclude<EmbeddingModel, EmbeddingModel.CUSTOM>,
    cacheDir: PathLike,
    showDownloadProgress: boolean = true
  ): Promise<PathLike> {
    const { repo, modelFile, additionalFiles = [] } = MODEL_SOURCES[model];
    const modelDir = path.join(cacheDir.toString(), repo.replace("/", "_"));

    // Files already present are skipped, so an interrupted download resumes with the missing files
    for (const fileName of [modelFile, ...additionalFiles, ...TOKENIZER_FILES]) {
      await this.downloadFileFromHF(
        path.join(modelDir, fileName),
        repo,
        fileName,
        showDownloadProgress
      );
    }
    return modelDir;
  }

  async *embed(textStrings: string[], batchSize: number = 256) {
    for (let i = 0; i < textStrings.length; i += batchSize) {
      const batchTexts = textStrings.slice(i, i + batchSize);

      const encodedTexts = await Promise.all(
        batchTexts.map((textString) => this.tokenizer.encode(textString))
      );

      const idsArray: bigint[][] = [];
      const maskArray: bigint[][] = [];
      const typeIdsArray: bigint[][] = [];

      encodedTexts.forEach((text) => {
        const ids = text.getIds().map(BigInt);
        const mask = text.getAttentionMask().map(BigInt);
        const typeIds = text.getTypeIds().map(BigInt);

        idsArray.push(ids);
        maskArray.push(mask);
        typeIdsArray.push(typeIds);
      });

      const maxLength = idsArray[0].length;

      const batchInputIds = new ort.Tensor(
        "int64",
        idsArray.flat() as unknown as number[],
        [batchTexts.length, maxLength]
      );
      const batchAttentionMask = new ort.Tensor(
        "int64",
        maskArray.flat() as unknown as number[],
        [batchTexts.length, maxLength]
      );
      const batchTokenTypeId = new ort.Tensor(
        "int64",
        typeIdsArray.flat() as unknown as number[],
        [batchTexts.length, maxLength]
      );

      const inputs: ModelInput = {
        input_ids: batchInputIds,
        attention_mask: batchAttentionMask,
        token_type_ids: batchTokenTypeId,
      };

      // Exclude token_type_ids for MLE5Large
      if (this.model === EmbeddingModel.MLE5Large) {
        delete inputs.token_type_ids;
      }

      const output = await this.session.run(inputs);

      const embeddings = poolEmbeddings(
        output.last_hidden_state.data as Float32Array,
        output.last_hidden_state.dims as [number, number, number],
        maskArray,
        this.pooling
      );

      yield embeddings.map(normalize);
    }
  }

  passageEmbed(texts: string[], batchSize: number = 256) {
    texts = texts.map((text) => `passage: ${text}`);
    return this.embed(texts, batchSize);
  }

  async queryEmbed(query: string): Promise<number[]> {
    return (await this.embed([`query: ${query}`]).next()).value![0];
  }

  listSupportedModels(): ModelInfo[] {
    return [
      {
        model: EmbeddingModel.BGESmallEN,
        dim: 384,
        description: "Fast English model",
      },
      {
        model: EmbeddingModel.BGESmallENV15,
        dim: 384,
        description: "v1.5 release of the fast, default English model",
      },
      {
        model: EmbeddingModel.BGEBaseEN,
        dim: 768,
        description: "Base English model",
      },
      {
        model: EmbeddingModel.BGEBaseENV15,
        dim: 768,
        description: "v1.5 release of Base English model",
      },
      {
        model: EmbeddingModel.BGESmallZH,
        dim: 512,
        description: "v1.5 release of the fast, Chinese model",
      },
      {
        model: EmbeddingModel.AllMiniLML6V2,
        dim: 384,
        description: "Sentence Transformer model, MiniLM-L6-v2",
      },
      {
        model: EmbeddingModel.MLE5Large,
        dim: 1024,
        description:
          "Multilingual model, e5-large. Recommend using this model for non-English languages",
      },
    ];
  }
}

// Sparse embedding implementation class
export class SparseTextEmbedding extends SparseEmbedding {
  private constructor(
    private tokenizer: Tokenizer,
    private session: ort.InferenceSession,
    private model: SparseEmbeddingModel,
    private vocabSize: number
  ) {
    super();
  }

  static async init(
    options: InitSparseStandardOptions
  ): Promise<SparseTextEmbedding>;
  static async init(
    options: InitSparseCustomOptions
  ): Promise<SparseTextEmbedding>;
  static async init({
    model = SparseEmbeddingModel.SpladePPEnV1,
    executionProviders = [ExecutionProvider.CPU],
    maxLength = 512,
    cacheDir = "local_cache",
    showDownloadProgress = true,
    modelAbsoluteDirPath = "",
    modelName = "",
  }: Partial<InitSparseOptions> = {}) {
    if (model === SparseEmbeddingModel.CUSTOM) {
      if (!modelAbsoluteDirPath) {
        throw new Error(
          "For custom model, modelAbsoluteDirPath is required in SparseTextEmbedding.init"
        );
      }
      if (!modelName) {
        throw new Error(
          "For custom model, modelName is required in SparseTextEmbedding.init"
        );
      }
    }

    const modelDir =
      model === SparseEmbeddingModel.CUSTOM
        ? modelAbsoluteDirPath
        : await SparseTextEmbedding.retrieveModel(
            model,
            cacheDir,
            showDownloadProgress
          );

    const { tokenizer, vocabSize } = this.loadTokenizer(modelDir, maxLength);

    const defaultModelName = "model.onnx";
    const modelPath = path.join(
      modelDir.toString(),
      "onnx",
      modelName || defaultModelName
    );

    if (!fs.existsSync(modelPath)) {
      throw new Error(`Model file not found at ${modelPath}`);
    }

    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders,
      graphOptimizationLevel: "all",
    });

    return new SparseTextEmbedding(tokenizer, session, model, vocabSize);
  }

  private static loadTokenizer(
    modelDir: fs.PathLike,
    maxLength: number
  ): { tokenizer: Tokenizer; vocabSize: number } {
    const tokenizerPath = path.join(modelDir.toString(), "tokenizer.json");
    if (!fs.existsSync(tokenizerPath)) {
      throw new Error(`Tokenizer file not found at ${tokenizerPath}`);
    }

    const configPath = path.join(modelDir.toString(), "config.json");
    if (!fs.existsSync(configPath)) {
      throw new Error(`Config file not found at ${configPath}`);
    }
    const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));

    const tokenizerFilePath = path.join(
      modelDir.toString(),
      "tokenizer_config.json"
    );
    if (!fs.existsSync(tokenizerFilePath)) {
      throw new Error(`Tokenizer file not found at ${tokenizerFilePath}`);
    }
    const tokenizerConfig = JSON.parse(
      fs.readFileSync(tokenizerFilePath, "utf-8")
    );
    maxLength = Math.min(maxLength, tokenizerConfig["model_max_length"]);

    const tokensMapPath = path.join(
      modelDir.toString(),
      "special_tokens_map.json"
    );
    if (!fs.existsSync(tokensMapPath)) {
      throw new Error(`Tokens map file not found at ${tokensMapPath}`);
    }
    const tokensMap = JSON.parse(fs.readFileSync(tokensMapPath, "utf-8"));

    const tokenizer = Tokenizer.fromFile(tokenizerPath);

    tokenizer.setTruncation(maxLength);
    tokenizer.setPadding({
      maxLength,
      padId: config["pad_token_id"],
      padToken: tokenizerConfig["pad_token"],
    });

    for (let token of Object.values(tokensMap)) {
      if (typeof token === "string") {
        tokenizer.addSpecialTokens([token]);
      } else if (isAddedTokenMap(token)) {
        const addedToken = new AddedToken(token["content"], true, {
          singleWord: token["single_word"],
          leftStrip: token["lstrip"],
          rightStrip: token["rstrip"],
          normalized: token["normalized"],
        });
        tokenizer.addAddedTokens([addedToken]);
      }
    }

    const vocabSize = config["vocab_size"] || 30522;

    return { tokenizer, vocabSize };
  }

  private static async retrieveModel(
    model: SparseEmbeddingModel,
    cacheDir: PathLike,
    showDownloadProgress: boolean = true
  ): Promise<PathLike> {
    if (!fs.existsSync(cacheDir)) {
      fs.mkdirSync(cacheDir, {
        mode: 0o777,
      });
    }

    const modelDir = path.join(cacheDir.toString(), model.replace("/", "_"));

    if (fs.existsSync(modelDir)) {
      return modelDir;
    }

    fs.mkdirSync(modelDir, { mode: 0o777 });

    // Download required files from hf
    const filesToDownload = [
      "onnx/model.onnx",
      "tokenizer.json",
      "tokenizer_config.json",
      "config.json",
      "special_tokens_map.json",
    ];

    for (const fileName of filesToDownload) {
      const outputPath = path.join(modelDir, fileName);
      const outputDir = path.dirname(outputPath);

      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true, mode: 0o777 });
      }

      // Use HuggingFace Hub library to download
      const downloaded = await downloadFileToCacheDir({
        repo: model,
        path: fileName,
      });

      // Copy from HF cache to our cache directory
      // In Node.js, downloadFile returns a string path
      if (downloaded && typeof downloaded === "string") {
        fs.copyFileSync(downloaded, outputPath);
      }
    }

    return modelDir;
  }

  async *embed(textStrings: string[], batchSize: number = 256) {
    for (let i = 0; i < textStrings.length; i += batchSize) {
      const batchTexts = textStrings.slice(i, i + batchSize);

      const encodedTexts = await Promise.all(
        batchTexts.map((textString) => this.tokenizer.encode(textString))
      );

      const idsArray: bigint[][] = [];
      const maskArray: number[][] = [];
      const typeIdsArray: bigint[][] = [];

      encodedTexts.forEach((text) => {
        const ids = text.getIds().map(BigInt);
        const mask = text.getAttentionMask();
        const typeIds = text.getTypeIds().map(BigInt);

        idsArray.push(ids);
        maskArray.push(mask);
        typeIdsArray.push(typeIds);
      });

      const maxLength = idsArray[0].length;

      const batchInputIds = new ort.Tensor(
        "int64",
        idsArray.flat() as unknown as number[],
        [batchTexts.length, maxLength]
      );
      const batchAttentionMask = new ort.Tensor(
        "int64",
        maskArray.flat().map(BigInt) as unknown as number[],
        [batchTexts.length, maxLength]
      );
      const batchTokenTypeId = new ort.Tensor(
        "int64",
        typeIdsArray.flat() as unknown as number[],
        [batchTexts.length, maxLength]
      );

      const inputs: ModelInput = {
        input_ids: batchInputIds,
        input_mask: batchAttentionMask,
        segment_ids: batchTokenTypeId,
      };

      const output = await this.session.run(inputs);

      // SPLADE postprocessing: log(1 + ReLU(logits))
      // @ts-expect-error this is incorrect it is there?
      const logits = output.output.cpuData as Float32Array;
      const dims = output.output.dims as [number, number, number];
      const [currentBatchSize, seqLen, vocabSize] = dims;

      const sparseVectors: SparseVector[] = [];

      for (let batchIdx = 0; batchIdx < currentBatchSize; batchIdx++) {
        const values = new Float32Array(vocabSize).fill(0);

        // Apply log(1 + ReLU(logits)) and max pooling
        for (let seqIdx = 0; seqIdx < seqLen; seqIdx++) {
          const attentionValue = maskArray[batchIdx][seqIdx];

          if (attentionValue > 0) {
            for (let vocabIdx = 0; vocabIdx < vocabSize; vocabIdx++) {
              const logitIdx =
                batchIdx * seqLen * vocabSize + seqIdx * vocabSize + vocabIdx;
              const logitValue = logits[logitIdx];

              // ReLU
              const reluValue = Math.max(0, logitValue);

              // log(1 + ReLU)
              const logValue = Math.log(1 + reluValue);

              // Max pooling over sequence
              values[vocabIdx] = Math.max(values[vocabIdx], logValue);
            }
          }
        }

        // Convert to sparse representation (only non-zero values)
        const sparseVector: SparseVector = {
          values: [],
          indices: []
        };
        for (let tokenId = 0; tokenId < vocabSize; tokenId++) {
          if (values[tokenId] > 0) {
            sparseVector.indices.push(tokenId)
            sparseVector.values.push(values[tokenId])
          }
        }
        sparseVectors.push(sparseVector);
      }

      yield sparseVectors;
    }
  }

  passageEmbed(texts: string[], batchSize: number = 256) {
    // SPLADE doesn't use passage/query prefixes like dense models
    return this.embed(texts, batchSize);
  }

  async queryEmbed(query: string): Promise<SparseVector> {
    return (await this.embed([query]).next()).value![0];
  }

  listSupportedModels(): SparseModelInfo[] {
    return [
      {
        model: SparseEmbeddingModel.SpladePPEnV1,
        vocabSize: 30522,
        description: "SPLADE++ English model for sparse retrieval",
      },
    ];
  }
}

interface AddedTokenMap {
  content: string;
  single_word: boolean;
  lstrip: boolean;
  rstrip: boolean;
  normalized: boolean;
}

function isAddedTokenMap(token: any): token is AddedTokenMap {
  return (
    typeof token === "object" &&
    token !== null &&
    "token" in token &&
    "single_word" in token &&
    "rstrip" in token &&
    "lstrip" in token &&
    "normalized" in token
  );
}
