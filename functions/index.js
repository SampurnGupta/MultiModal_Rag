require("dotenv").config();

const { onRequest } = require("firebase-functions/v2/https");
const admin = require("firebase-admin");
const { GoogleGenerativeAI } = require("@google/generative-ai");

if (!admin.apps.length) {
  admin.initializeApp();
}
const db = admin.firestore();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
  console.warn("⚠️ GEMINI_API_KEY is not set in .env");
}

// Cosine similarity between two vectors
function cosineSimilarity(a, b) {
  const len = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (!na || !nb) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// Chunk long text into smaller pieces
function chunkText(text, maxChars = 800) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    chunks.push(text.slice(start, start + maxChars));
    start += maxChars;
  }
  return chunks;
}

// Create Gemini client + models
function getGeminiClients() {
  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
  const chatModel = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
  const embeddingModel = genAI.getGenerativeModel({
    model: "text-embedding-004",
  });
  return { chatModel, embeddingModel };
}

// ---- 1) Ingest text (from docs / OCR / audio) ----
// Expects: { text: "...", sourceType: "doc|image|audio", sourceName?: string }
exports.ingestText = onRequest(async (req, res) => {
  // Basic CORS
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return;
  }
  if (req.method !== "POST") {
    res.status(405).send("Use POST");
    return;
  }

  const { text, sourceType = "doc", sourceName = "unknown" } = req.body || {};
  if (!text || !text.trim()) {
    res.status(400).send("Missing text");
    return;
  }

  const { embeddingModel } = getGeminiClients();

  try {
    const chunks = chunkText(text);

    // ✅ NEW API: use `requests`, not `contents`
    const batchRes = await embeddingModel.batchEmbedContents({
      requests: chunks.map((c) => ({
        content: {
          role: "user",
          parts: [{ text: c }],
        },
      })),
    });

    const embeddings = batchRes.embeddings || [];
    const batch = db.batch();
    const col = db.collection("rag_chunks");

    chunks.forEach((chunk, i) => {
      const emb = embeddings[i]?.values || [];
      const docRef = col.doc();
      batch.set(docRef, {
        text: chunk,
        embedding: emb,
        sourceType,
        sourceName,
        createdAt: Date.now(), // plain JS timestamp
      });
    });

    await batch.commit();
    res.json({ ok: true, chunks: chunks.length });
  } catch (err) {
    console.error("Error in ingestText:", err);
    res.status(500).send("Error ingesting text");
  }
});

// ---- 2) Chat with RAG ----
// Expects: { message: "question..." }
exports.chatRag = onRequest(async (req, res) => {
  // Basic CORS
  res.set("Access-Control-Allow-Origin", "*");
  res.set("Access-Control-Allow-Headers", "Content-Type");
  if (req.method === "OPTIONS") {
    res.status(204).send("");
    return;
  }
  if (req.method !== "POST") {
    res.status(405).send("Use POST");
    return;
  }

  const { message } = req.body || {};
  if (!message || !message.trim()) {
    res.status(400).send("Missing message");
    return;
  }

  const { chatModel, embeddingModel } = getGeminiClients();

  try {
    const normalized = message.trim().toLowerCase();
    const wordCount = normalized.split(/\s+/).filter(Boolean).length;
    const isGreeting = /^(hi+|hello+|hey+|yo+|sup|good (morning|evening|afternoon|night))\b/.test(
      normalized
    );

    // 1) Handle pure greetings WITHOUT RAG so you don't get weird facts for "hi"
    if (isGreeting && wordCount <= 5) {
      const resp = await chatModel.generateContent({
        contents: [
          {
            role: "user",
            parts: [
              {
                text: `The user greeted you with: "${message}". 
Reply with a short, friendly greeting (1–2 sentences) and briefly explain that you are a multimodal RAG chatbot that can answer questions about their uploaded text, images (via OCR), and audio transcripts.`,
              },
            ],
          },
        ],
      });

      const text =
        resp.response?.candidates?.[0]?.content?.parts?.[0]?.text ||
        "Hi! I’m your multimodal RAG assistant. You can upload text, images or audio and then ask questions about them.";
      res.json({ answer: text, topChunks: [] });
      return;
    }

    // 2) Normal RAG path
    // Embed the user question
    const embRes = await embeddingModel.embedContent({
      content: {
        role: "user",
        parts: [{ text: message }],
      },
    });
    const queryEmbedding = embRes.embedding.values;

    // Fetch all chunks from Firestore (fine for small demo)
    const snap = await db.collection("rag_chunks").get();
    const docs = [];
    snap.forEach((d) => docs.push({ id: d.id, ...d.data() }));

    // Compute similarity & pick top K
    const scored = docs.map((d) => ({
      ...d,
      score: Array.isArray(d.embedding)
        ? cosineSimilarity(queryEmbedding, d.embedding)
        : 0,
    }));
    scored.sort((a, b) => b.score - a.score);
    const topK = scored.slice(0, 5);

    const contextText = topK
      .map(
        (d, i) =>
          `[#${i + 1} | ${d.sourceType}:${d.sourceName} | score=${d.score.toFixed(
            3
          )}]\n${d.text}`
      )
      .join("\n\n");

    const prompt = `
You are a helpful assistant that answers questions about the user's uploaded content (documents, OCR text from images, and audio transcripts).

Use the "Context" below to ground your answer:
- Prefer information that clearly matches the user's question.
- If the answer is not clearly supported by the context, say you don't know.
- Write in a natural, conversational tone.
- Keep answers concise (1–3 sentences).

User question:
${message}

Context:
${contextText}
`;

    const resp = await chatModel.generateContent({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
    });

    const text =
      resp.response?.candidates?.[0]?.content?.parts?.[0]?.text ||
      "(no answer)";

    res.json({
      answer: text,
      topChunks: topK.map((d) => ({
        id: d.id,
        sourceType: d.sourceType,
        sourceName: d.sourceName,
        score: d.score,
      })),
    });
  } catch (err) {
    console.error("Error in chatRag:", err);
    res.status(500).send("Error generating answer");
  }
});
