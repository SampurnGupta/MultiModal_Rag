// Add this file to functions/server.js
// (No github URL because this is a new file to add)
require("dotenv").config();

const express = require("express");
const path = require("path");
const admin = require("firebase-admin");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const cors = require("cors");

// Initialize Firebase Admin:
// If you provide FIREBASE_SERVICE_ACCOUNT_JSON in Render env, use that.
// Otherwise try default application credentials (may not work on Render).
if (!admin.apps.length) {
  if (process.env.FIREBASE_SERVICE_ACCOUNT_JSON) {
    try {
      const sa = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT_JSON);
      admin.initializeApp({
        credential: admin.credential.cert(sa),
        // optional: databaseURL: process.env.FIREBASE_DATABASE_URL
      });
      console.log("Initialized firebase-admin from FIREBASE_SERVICE_ACCOUNT_JSON");
    } catch (err) {
      console.error("Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON:", err);
      // fallback to default
      admin.initializeApp();
    }
  } else {
    admin.initializeApp();
    console.log("Initialized firebase-admin with default credentials");
  }
}
const db = admin.firestore();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
  console.warn("⚠️ GEMINI_API_KEY is not set in environment");
}

// Cosine similarity
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

// Chunk long text
function chunkText(text, maxChars = 800) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    chunks.push(text.slice(start, start + maxChars));
    start += maxChars;
  }
  return chunks;
}

// Gemini helpers
function getGeminiClients() {
  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
  const chatModel = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
  const embeddingModel = genAI.getGenerativeModel({
    model: "text-embedding-004",
  });
  return { chatModel, embeddingModel };
}

const app = express();
app.use(cors()); // allow all origins by default; tighten later if needed
app.use(express.json({ limit: "10mb" }));

// ---- 1) Ingest text: POST /ingestText ----
app.post("/ingestText", async (req, res) => {
  const { text, sourceType = "doc", sourceName = "unknown" } = req.body || {};
  if (!text || !text.trim()) {
    res.status(400).send("Missing text");
    return;
  }

  const { embeddingModel } = getGeminiClients();

  try {
    const chunks = chunkText(text);

    // Use batch embed API like in the original code
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
        createdAt: Date.now(),
      });
    });

    await batch.commit();
    res.json({ ok: true, chunks: chunks.length });
  } catch (err) {
    console.error("Error in /ingestText:", err);
    res.status(500).send("Error ingesting text");
  }
});

// ---- 2) Chat with RAG: POST /chatRag ----
app.post("/chatRag", async (req, res) => {
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

    // greeting shortcut
    if (isGreeting && wordCount <= 5) {
      const resp = await chatModel.generateContent({
        contents: [
          {
            role: "user",
            parts: [
              {
                text: `The user greeted you with: "${message}". Reply with a short, friendly greeting (1–2 sentences) and briefly explain that you are a multimodal RAG chatbot.`,
              },
            ],
          },
        ],
      });

      const text =
        resp.response?.candidates?.[0]?.content?.parts?.[0]?.text ||
        "Hi! I’m your multimodal RAG assistant.";
      res.json({ answer: text, topChunks: [] });
      return;
    }

    // embed the question
    const embRes = await embeddingModel.embedContent({
      content: {
        role: "user",
        parts: [{ text: message }],
      },
    });
    const queryEmbedding = embRes.embedding.values;

    // fetch chunks (small demo; may need paging for large datasets)
    const snap = await db.collection("rag_chunks").get();
    const docs = [];
    snap.forEach((d) => docs.push({ id: d.id, ...d.data() }));

    const scored = docs.map((d) => ({
      ...d,
      score: Array.isArray(d.embedding) ? cosineSimilarity(queryEmbedding, d.embedding) : 0,
    }));
    scored.sort((a, b) => b.score - a.score);
    const topK = scored.slice(0, 5);

    const contextText = topK
      .map(
        (d, i) =>
          `[#${i + 1} | ${d.sourceType}:${d.sourceName} | score=${d.score.toFixed(3)}]\n${d.text}`
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
      resp.response?.candidates?.[0]?.content?.parts?.[0]?.text || "(no answer)";

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
    console.error("Error in /chatRag:", err);
    res.status(500).send("Error generating answer");
  }
});

// Serve static frontend from ../public
const publicPath = path.join(__dirname, "..", "public");
app.use(express.static(publicPath));

// SPA fallback - send index.html for unknown routes (optional)
app.get("*", (req, res) => {
  const indexFile = path.join(publicPath, "index.html");
  if (require("fs").existsSync(indexFile)) {
    res.sendFile(indexFile);
  } else {
    res.status(404).send("Not found");
  }
});

// Start server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});