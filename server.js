const express = require("express");
const cors = require("cors");
const path = require("path");
const { GoogleGenerativeAI } = require("@google/generative-ai");

// --- Config ---
const PORT = process.env.PORT || 3000;
const GEMINI_API_KEY =
  process.env.GEMINI_API_KEY; // replace for local tests if you want

if (!GEMINI_API_KEY || GEMINI_API_KEY.length < 20) {
  console.warn("⚠️ GEMINI_API_KEY is not configured. Set it in env for production.");
}

// --- Gemini setup ---
const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
const chatModel = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
const embeddingModel = genAI.getGenerativeModel({
  model: "text-embedding-004",
});

// --- In-memory store for RAG chunks ---
/**
 * Each entry:
 * {
 *   id: string,
 *   text: string,
 *   embedding: number[],
 *   sourceType: "doc" | "image" | "audio",
 *   sourceName: string,
 *   createdAt: number
 * }
 */
const ragChunks = [];
let nextId = 1;

// --- Helpers ---
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

function chunkText(text, maxChars) {
  maxChars = maxChars || 800;
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    chunks.push(text.slice(start, start + maxChars));
    start += maxChars;
  }
  return chunks;
}

// --- Express app ---
const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));

// Serve static frontend from ./public
const publicDir = path.join(__dirname, "public");
app.use(express.static(publicDir));

// --- API endpoints (match old Firebase ones) ---

// 1) Ingest text (from docs / OCR / audio)
app.post("/ingestText", async (req, res) => {
  try {
    const body = req.body || {};
    const text = body.text;
    const sourceType = body.sourceType || "doc";
    const sourceName = body.sourceName || "unknown";

    if (!text || !String(text).trim()) {
      return res.status(400).send("Missing text");
    }

    if (!GEMINI_API_KEY || GEMINI_API_KEY.length < 20) {
      return res
        .status(500)
        .send("GEMINI_API_KEY is not configured on the server");
    }

    const chunks = chunkText(text);
    const requests = chunks.map((c) => ({
      content: {
        role: "user",
        parts: [{ text: c }],
      },
    }));

    const batchRes = await embeddingModel.batchEmbedContents({
      requests,
    });

    const embeddings = batchRes.embeddings || [];

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const emb =
        embeddings[i] && Array.isArray(embeddings[i].values)
          ? embeddings[i].values
          : [];
      ragChunks.push({
        id: String(nextId++),
        text: chunk,
        embedding: emb,
        sourceType,
        sourceName,
        createdAt: Date.now(),
      });
    }

    res.json({ ok: true, chunks: chunks.length });
  } catch (err) {
    console.error("Error in /ingestText:", err);
    res
      .status(500)
      .send(
        "Error ingesting text on server: " + (err.message || "unknown error")
      );
  }
});

// 2) Chat with RAG
app.post("/chatRag", async (req, res) => {
  try {
    const body = req.body || {};
    const message = body.message;
    if (!message || !String(message).trim()) {
      return res.status(400).send("Missing message");
    }

    if (!GEMINI_API_KEY || GEMINI_API_KEY.length < 20) {
      return res
        .status(500)
        .send("GEMINI_API_KEY is not configured on the server");
    }

    const normalized = message.trim().toLowerCase();
    const words = normalized.split(/\s+/).filter(Boolean);
    const wordCount = words.length;
    const greetingRegex =
      /^(hi+|hello+|hey+|yo+|sup|good (morning|evening|afternoon|night))\b/;
    const isGreeting = greetingRegex.test(normalized);

    // 1) Greeting path (no RAG)
    if (isGreeting && wordCount <= 5) {
      const greetPrompt =
        'The user greeted you with: "' +
        message +
        '". ' +
        "Reply with a short, friendly greeting (1–2 sentences) and briefly explain that you are a multimodal RAG chatbot that can answer questions about their uploaded text, images (via OCR), and audio transcripts.";

      const greetResp = await chatModel.generateContent({
        contents: [
          {
            role: "user",
            parts: [{ text: greetPrompt }],
          },
        ],
      });

      let answer =
        "Hi! I’m your multimodal RAG assistant. You can upload text, images or audio and then ask questions about them.";
      if (
        greetResp &&
        greetResp.response &&
        greetResp.response.candidates &&
        greetResp.response.candidates.length > 0
      ) {
        const cand = greetResp.response.candidates[0];
        if (
          cand.content &&
          cand.content.parts &&
          cand.content.parts.length > 0 &&
          cand.content.parts[0].text
        ) {
          answer = cand.content.parts[0].text;
        }
      }

      return res.json({ answer, topChunks: [] });
    }

    // 2) Normal RAG path
    const embRes = await embeddingModel.embedContent({
      content: {
        role: "user",
        parts: [{ text: message }],
      },
    });

    const queryEmbedding =
      embRes &&
      embRes.embedding &&
      Array.isArray(embRes.embedding.values)
        ? embRes.embedding.values
        : [];

    // score against stored chunks
    const scored = ragChunks.map((d) => {
      let score = 0;
      if (d.embedding && Array.isArray(d.embedding)) {
        score = cosineSimilarity(queryEmbedding, d.embedding);
      }
      return {
        id: d.id,
        text: d.text,
        sourceType: d.sourceType,
        sourceName: d.sourceName,
        score,
      };
    });

    scored.sort((a, b) => b.score - a.score);
    const topK = scored.slice(0, 5);

    let contextText = "";
    for (let i = 0; i < topK.length; i++) {
      const d = topK[i];
      contextText +=
        "[#" +
        (i + 1) +
        " | " +
        d.sourceType +
        ":" +
        d.sourceName +
        " | score=" +
        d.score.toFixed(3) +
        "]\n" +
        d.text +
        "\n\n";
    }

    const ragPrompt =
      "You are a helpful assistant that answers questions about the user's uploaded content (documents, OCR text from images, and audio transcripts).\n\n" +
      'Use the "Context" below to ground your answer:\n' +
      "- Prefer information that clearly matches the user's question.\n" +
      "- If the answer is not clearly supported by the context, say you don't know.\n" +
      "- Write in a natural, conversational tone.\n" +
      "- Keep answers concise (1–3 sentences).\n\n" +
      "User question:\n" +
      message +
      "\n\nContext:\n" +
      contextText;

    const ragResp = await chatModel.generateContent({
      contents: [
        {
          role: "user",
          parts: [{ text: ragPrompt }],
        },
      ],
    });

    let finalText = "(no answer)";
    if (
      ragResp &&
      ragResp.response &&
      ragResp.response.candidates &&
      ragResp.response.candidates.length > 0
    ) {
      const cand = ragResp.response.candidates[0];
      if (
        cand.content &&
        cand.content.parts &&
        cand.content.parts.length > 0 &&
        cand.content.parts[0].text
      ) {
        finalText = cand.content.parts[0].text;
      }
    }

    res.json({
      answer: finalText,
      topChunks: topK.map((d) => ({
        id: d.id,
        sourceType: d.sourceType,
        sourceName: d.sourceName,
        score: d.score,
      })),
    });
  } catch (err) {
    console.error("Error in /chatRag:", err);
    res
      .status(500)
      .send("Error generating answer: " + (err.message || "unknown error"));
  }
});

// --- Fallback: send index.html for root ---
// Catch-all for SPA — must come AFTER all API routes
app.use((req, res) => {
  res.sendFile(path.join(publicDir, "index.html"));
});

app.listen(PORT, () => {
  console.log("🚀 Server running on port", PORT);
});
