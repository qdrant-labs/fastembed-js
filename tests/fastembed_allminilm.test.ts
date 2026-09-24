import { expect, test } from "vitest";
import { FlagEmbedding, EmbeddingModel } from "../src";

test('Init EmbeddingModel', async () => {
    const model = await FlagEmbedding.init({
        model: EmbeddingModel.AllMiniLML6V2
    });
    expect(model).toBeDefined();
});

test("FlagEmbedding embed", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,
    maxLength: 512,
  });
  const embeddings = (await flagEmbedding.embed(["This is a test"]).next())
    .value!;
  expect(embeddings).toBeDefined();
  expect(embeddings.length).toBe(1);
});

test("FlagEmbedding embed batch", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,

    maxLength: 512,
  });
  const embeddingsBatch = flagEmbedding.embed(["This is a test", "Some text", "Some more test", "This is a test", "Some text", "Some more test"]);
  for await (const embeddings of embeddingsBatch) {
    expect(embeddings).toBeDefined();
    expect(embeddings.length).toBe(6);
    expect(embeddings[0].length).toBe(384);
  }
});

test("FlagEmbedding embed small batch", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,
    maxLength: 512,
  });
  const embeddingsBatch = flagEmbedding.embed(["This is a test", "Some text"], 1);
  for await (const embeddings of embeddingsBatch) {
    expect(embeddings).toBeDefined();
    expect(embeddings.length).toBe(1);
    expect(embeddings[0].length).toBe(384);
  }
});

test("FlagEmbedding queryEmbed", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,
    maxLength: 512,
  });
  const embeddings = await flagEmbedding.queryEmbed("This is a test");
  expect(embeddings).toBeDefined();
  expect(embeddings.length).toBe(384);
});

test("FlagEmbedding passageEmbed", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,

    maxLength: 512,
  });
  const embeddings = (
    await flagEmbedding.passageEmbed(["This is a test"]).next()
  ).value!;
  expect(embeddings).toBeDefined();
  expect(embeddings.length).toBe(1);
});

test("FlagEmbedding canonical values", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,
    maxLength: 512,
  });
  const expected = [
    -0.03448, 0.03102, 0.00673, 0.02611, -0.03936, -0.1603, 0.06692, -0.00644,
    -0.04745, 0.01476,
  ];

  const embeddings = (await flagEmbedding.embed(["hello world"]).next()).value!;
  expect(embeddings).toBeDefined();
  for (let i = 0; i < expected.length; i++) {
    expect(embeddings[0][i]).toBeCloseTo(expected[i], 3);
  }
});
