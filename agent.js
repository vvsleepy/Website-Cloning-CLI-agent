/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║           Scaler CLI Agent — General-Purpose AI Assistant       ║
 * ║     Assignment 02 | GenAI | ReAct Loop (THINK→TOOL→OBSERVE)     ║
 * ╚══════════════════════════════════════════════════════════════════╝
 *
 * A conversational CLI agent powered by Google Gemini 2.5 Flash.
 * Works exactly like Cursor / Windsurf — accepts ANY natural language
 * instruction and acts on it through a strict ReAct reasoning loop.
 *
 * Example tasks the user can give:
 *   • "What's the weather in Mumbai?"
 *   • "Show me GitHub info for torvalds"
 *   • "Create a todo app using HTML, CSS, and JS"
 *   • "Clone the Scaler Academy website with Header, Hero, and Footer"
 *   • "List files in this directory and tell me the largest one"
 */

import "dotenv/config";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { OpenAI } from "openai";
import { exec } from "child_process";
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { fileURLToPath } from "url";
import { promisify } from "util";
import open from "open";
import axios from "axios";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const execAsync = promisify(exec);

// ─── Provider Setup ───────────────────────────────────────────────────────────
// Supports Groq, OpenRouter, and Gemini simultaneously.
// All three keys can be in .env at once — if one provider is exhausted it
// automatically falls through to the next. No manual switching needed.
//
// .env:
//   GROQ_API_KEY=gsk_...
//   OPENROUTER_API_KEY=sk-or-...
//   GEMINI_API_KEY=AIza...

const GROQ_API_KEY       = process.env.GROQ_API_KEY;
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
const GEMINI_API_KEY     = process.env.GEMINI_API_KEY;

if (!GROQ_API_KEY && !OPENROUTER_API_KEY && !GEMINI_API_KEY) {
  console.error("\x1b[31m❌  No API key found in .env\x1b[0m");
  console.error("  GROQ_API_KEY=gsk_...          (free at console.groq.com)");
  console.error("  OPENROUTER_API_KEY=sk-or-...  (free at openrouter.ai)");
  console.error("  GEMINI_API_KEY=AIza...         (free at aistudio.google.com)");
  process.exit(1);
}

// ── Clients ──────────────────────────────────────────────────────────────────
const groqClient = GROQ_API_KEY ? new OpenAI({
  apiKey: GROQ_API_KEY,
  baseURL: "https://api.groq.com/openai/v1",
  timeout: 25000, maxRetries: 0,
}) : null;

const orClient = OPENROUTER_API_KEY ? new OpenAI({
  apiKey: OPENROUTER_API_KEY,
  baseURL: "https://openrouter.ai/api/v1",
  timeout: 25000, maxRetries: 0,
  defaultHeaders: { "HTTP-Referer": "https://github.com/scaler-cli-agent", "X-Title": "Scaler CLI Agent" },
}) : null;

const genAI = GEMINI_API_KEY ? new GoogleGenerativeAI(GEMINI_API_KEY) : null;
const GEN_CONFIG = { temperature: 0.5, maxOutputTokens: 65536 };

// Build a flat ordered list of every available model across all providers.
// If all three keys are present: tries Groq → OpenRouter → Gemini in order.
const MODEL_QUEUE = [
  ...(groqClient ? [
    { label: "Groq/llama-3.1-8b",    call: (p) => groqClient.chat.completions.create({ model: "llama-3.1-8b-instant",    messages: [{ role: "user", content: p }], temperature: 0.5, max_tokens: 4096 }).then(r => r.choices[0].message.content.trim()) },
    { label: "Groq/llama-3.3-70b",   call: (p) => groqClient.chat.completions.create({ model: "llama-3.3-70b-versatile", messages: [{ role: "user", content: p }], temperature: 0.5, max_tokens: 4096 }).then(r => r.choices[0].message.content.trim()) },
  ] : []),
  ...(orClient ? [
    { label: "OpenRouter/llama-3.3", call: (p) => orClient.chat.completions.create({ model: "meta-llama/llama-3.3-70b-instruct:free", messages: [{ role: "user", content: p }], temperature: 0.5, max_tokens: 4096 }).then(r => r.choices[0].message.content.trim()) },
    { label: "OpenRouter/mistral-7b", call: (p) => orClient.chat.completions.create({ model: "mistralai/mistral-7b-instruct:free",         messages: [{ role: "user", content: p }], temperature: 0.5, max_tokens: 4096 }).then(r => r.choices[0].message.content.trim()) },
  ] : []),
  ...(genAI ? [
    { label: "Gemini/2.0-flash",      call: (p) => genAI.getGenerativeModel({ model: "gemini-2.0-flash",      generationConfig: GEN_CONFIG }).generateContent(p).then(r => r.response.text().trim()) },
    { label: "Gemini/2.0-flash-lite", call: (p) => genAI.getGenerativeModel({ model: "gemini-2.0-flash-lite", generationConfig: GEN_CONFIG }).generateContent(p).then(r => r.response.text().trim()) },
    { label: "Gemini/2.5-flash",      call: (p) => genAI.getGenerativeModel({ model: "gemini-2.5-flash",      generationConfig: GEN_CONFIG }).generateContent(p).then(r => r.response.text().trim()) },
  ] : []),
];

const activeProviders = [groqClient && "Groq", orClient && "OpenRouter", genAI && "Gemini"].filter(Boolean).join(" + ");
console.log(`\x1b[90mProviders: ${activeProviders} (${MODEL_QUEUE.length} models in queue)\x1b[0m`);

/** Try every model in order. Skip instantly on quota/rate errors. */
async function callModel(prompt) {
  for (const { label, call } of MODEL_QUEUE) {
    try {
      const result = await call(prompt);
      return result;
    } catch (err) {
      const msg = err.message || "";
      const isSkip = msg.includes("429") || msg.includes("quota") || msg.includes("rate limit") ||
                      msg.includes("RESOURCE_EXHAUSTED") || msg.includes("503") || msg.includes("overloaded") ||
                      msg.includes("404") || msg.includes("No endpoints") || msg.includes("not found");
      if (isSkip) {
        console.log(clr(C.yellow, `⚡  ${label} unavailable — trying next...`));
        continue;
      }
      throw err; // auth error, network issue, etc — don't swallow it
    }
  }
  throw new Error("All models across all providers are exhausted. Wait a minute and try again.");
}

// ─── ANSI Colour Helpers ──────────────────────────────────────────────────────

const C = {
  reset:   "\x1b[0m",
  bold:    "\x1b[1m",
  dim:     "\x1b[2m",
  cyan:    "\x1b[36m",
  yellow:  "\x1b[33m",
  green:   "\x1b[32m",
  magenta: "\x1b[35m",
  red:     "\x1b[31m",
  blue:    "\x1b[34m",
  gray:    "\x1b[90m",
};
const clr = (color, text) => `${color}${text}${C.reset}`;

// ─── Tool: getWeather ─────────────────────────────────────────────────────────
/**
 * Fetch live weather for any city using wttr.in (no API key needed).
 */
async function getWeather({ city }) {
  try {
    const url = `https://wttr.in/${encodeURIComponent(city)}?format=%C+%t+%h+%w`;
    const { data } = await axios.get(url, { responseType: "text", timeout: 8000 });
    return `Weather in ${city}: ${data.trim()}`;
  } catch (err) {
    return `ERROR fetching weather: ${err.message}`;
  }
}

// ─── Tool: getGitHubUser ──────────────────────────────────────────────────────
/**
 * Fetch public GitHub profile info for any username.
 */
async function getGitHubUser({ username }) {
  try {
    const { data } = await axios.get(`https://api.github.com/users/${username}`, {
      timeout: 8000,
      headers: { "User-Agent": "ScalerCLIAgent/1.0" },
    });
    return {
      login:        data.login,
      name:         data.name,
      bio:          data.bio,
      location:     data.location,
      public_repos: data.public_repos,
      followers:    data.followers,
      following:    data.following,
      blog:         data.blog,
      created_at:   data.created_at,
    };
  } catch (err) {
    return `ERROR fetching GitHub user: ${err.message}`;
  }
}

// ─── Tool: executeCommand ─────────────────────────────────────────────────────
/**
 * Run any shell command. Accepts { cmd: string } or a plain string.
 */
async function executeCommand(args) {
  const cmd = typeof args === "string" ? args : (args?.cmd || "");
  console.log(clr(C.gray, `  ▶ Running: ${cmd}`));
  try {
    const { stdout, stderr } = await execAsync(cmd, {
      maxBuffer: 1024 * 1024 * 10,
      cwd: __dirname,
    });
    return stdout || stderr || "Command executed successfully.";
  } catch (err) {
    return `ERROR: ${err.message}`;
  }
}

// ─── Tool: writeFile ──────────────────────────────────────────────────────────
/**
 * Write (or overwrite) content to a file. Creates parent dirs automatically.
 */
async function writeFile({ filepath, content }) {
  const absPath = path.resolve(__dirname, filepath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.writeFileSync(absPath, content, "utf-8");
  return `File written: ${absPath} (${content.length} bytes)`;
}

// ─── Tool: appendFile ─────────────────────────────────────────────────────────
/**
 * Append content to an existing file (useful for writing large files in chunks).
 */
async function appendFile({ filepath, content }) {
  const absPath = path.resolve(__dirname, filepath);
  fs.mkdirSync(path.dirname(absPath), { recursive: true });
  fs.appendFileSync(absPath, content, "utf-8");
  const size = fs.statSync(absPath).size;
  return `Appended to: ${absPath} (total size: ${size} bytes)`;
}

// ─── Tool: readFile ───────────────────────────────────────────────────────────
/**
 * Read and return the contents of a file.
 */
async function readFile({ filepath }) {
  const absPath = path.resolve(__dirname, filepath);
  if (!fs.existsSync(absPath)) return `ERROR: File not found: ${absPath}`;
  const content = fs.readFileSync(absPath, "utf-8");
  return content.length > 4000 ? content.slice(0, 4000) + `\n... [truncated, total ${content.length} bytes]` : content;
}

// ─── Tool: listFiles ──────────────────────────────────────────────────────────
/**
 * List files and directories inside a given path.
 */
async function listFiles({ dirpath }) {
  const absPath = path.resolve(__dirname, dirpath || ".");
  if (!fs.existsSync(absPath)) return `ERROR: Directory not found: ${absPath}`;
  const entries = fs.readdirSync(absPath, { withFileTypes: true });
  if (entries.length === 0) return "(empty directory)";
  return entries
    .map((e) => (e.isDirectory() ? `📁 ${e.name}/` : `📄 ${e.name}`))
    .join("\n");
}

// ─── Tool: openInBrowser ──────────────────────────────────────────────────────
/**
 * Open a local HTML file (or URL) in the system's default browser.
 */
async function openInBrowser({ filepath }) {
  const absPath = path.resolve(__dirname, filepath);
  if (!fs.existsSync(absPath)) return `ERROR: File not found: ${absPath}`;
  await open(absPath);
  return `Opened in browser: ${absPath}`;
}

// ─── Tool: fetchWebpage ───────────────────────────────────────────────────────
async function fetchWebpage({ url }) {
  try {
    const res = await axios.get(`https://r.jina.ai/${url}`);
    return res.data;
  } catch (err) {
    return `Failed to fetch webpage: ${err.message}`;
  }
}

// ─── Tool Registry ────────────────────────────────────────────────────────────

const TOOL_MAP = {
  getWeather,
  getGitHubUser,
  executeCommand,
  writeFile,
  appendFile,
  readFile,
  listFiles,
  openInBrowser,
  fetchWebpage,
};

// ─── System Prompt ────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `
You are a general-purpose AI CLI Agent (like Cursor/Windsurf). You follow the ReAct loop:
THINK → TOOL → OBSERVE → (repeat) → OUTPUT.

TOOLS available (call with JSON "tool_args"):
  getWeather({city})            — live weather via wttr.in
  getGitHubUser({username})     — public GitHub profile
  executeCommand({cmd})         — run a shell command
  writeFile({filepath,content}) — write/overwrite a file (creates dirs)
  appendFile({filepath,content})— append to a file (use for large files in chunks)
  readFile({filepath})          — read a file
  listFiles({dirpath})          — list directory contents
  openInBrowser({filepath})     — open local file in browser
  fetchWebpage({url})           — fetches clean text/markdown from a live URL

RULES (follow strictly):
1. ONE JSON object per response — nothing outside JSON.
2. One step at a time. Wait for OBSERVE before next TOOL.
3. For simple factual/math questions: answer directly with OUTPUT — no tools needed.
4. To read a live website, ALWAYS use fetchWebpage({url}). NEVER use executeCommand with curl/wget.
5. HTML files: all CSS in <style>, all JS in <script>. Google Fonts via @import only.
6. Write files in a single writeFile call whenever possible.
7. After writing a file: verify with listFiles, then open with openInBrowser.
8. End with OUTPUT when the task is fully done.

WHEN ASKED TO CLONE OR RECREATE A WEBSITE:
- Step 1: ALWAYS use fetchWebpage({url}) to read the real website content first.
- Step 2: Write the ENTIRE clone in ONE SINGLE writeFile call. Do NOT split it.
- Step 3: Write a highly styled, professional, modern landing page.
- Step 4: Use flexbox/grid for layouts. Add rich CSS colors, padding, and hover effects in a <style> block.
- DO NOT output bare HTML skeletons. Make it look like a real tech startup.

OUTPUT FORMAT:
{"step":"THINK","content":"..."}
{"step":"TOOL","tool_name":"...","tool_args":{...}}
{"step":"OUTPUT","content":"..."}
`.trim();

// ─── Robust JSON Parser ───────────────────────────────────────────────────────

function parseModelResponse(rawText) {
  let cleaned = rawText.trim();

  // Strip markdown fences
  cleaned = cleaned.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/i, "").trim();

  // Extract all top-level JSON objects
  const jsonMatches = [];
  let depth = 0;
  let start = -1;
  let inString = false;
  let escape = false;

  for (let i = 0; i < cleaned.length; i++) {
    const ch = cleaned[i];
    if (escape) { escape = false; continue; }
    if (ch === "\\" && inString) { escape = true; continue; }
    if (ch === '"') { inString = !inString; continue; }
    if (inString) continue;

    if (ch === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === "}") {
      depth--;
      if (depth === 0 && start !== -1) {
        jsonMatches.push(cleaned.slice(start, i + 1));
        start = -1;
      }
    }
  }

  // Try parsing matches from last to first (prefer the last valid one)
  for (let i = jsonMatches.length - 1; i >= 0; i--) {
    try { return JSON.parse(jsonMatches[i]); } catch (_) { /* next */ }
  }

  // Last resort: parse the entire cleaned string
  return JSON.parse(cleaned);
}

// ─── Agent Loop ───────────────────────────────────────────────────────────────

async function runAgent(userMessage) {
  console.log(clr(C.cyan, "\n🤖 Agent started...\n"));

  // We maintain a running transcript and rebuild the full prompt each turn.
  // This is more reliable than Gemini's chat history API for our use-case.
  const transcript = [];
  let iteration = 0;
  const MAX_ITERATIONS = 60;

  while (iteration < MAX_ITERATIONS) {
    iteration++;

    // Add a small delay to avoid hitting rate limits (e.g., Requests Per Minute)
    if (iteration > 1) {
      await new Promise(r => setTimeout(r, 1500));
    }

    // ── Build full prompt (cap transcript to last 6 entries to avoid token blowup) ──
    let fullPrompt = `${SYSTEM_PROMPT}\n\nUser instruction: ${userMessage}\n`;
    const recentTranscript = transcript.slice(-6);
    for (const entry of recentTranscript) {
      if (entry.role === "assistant") {
        fullPrompt += `\nAssistant: ${entry.content}`;
      } else if (entry.role === "observe") {
        fullPrompt += `\n${JSON.stringify({ step: "OBSERVE", content: entry.content })}`;
      } else if (entry.role === "system") {
        fullPrompt += `\nSystem note: ${entry.content}`;
      }
    }
    fullPrompt += "\n\nRespond with the NEXT single JSON step object:";

    // ── Call Gemini (with retry + model fallback) ──
    let raw;
    try {
      raw = await callModel(fullPrompt);
    } catch (err) {
      console.error(clr(C.red, `❌  API error: ${err.message.slice(0, 120)}`));
      break;
    }

    // ── Parse JSON ──
    let parsed;
    try {
      parsed = parseModelResponse(raw);
    } catch (_) {
      console.error(clr(C.yellow, "⚠️  Non-JSON response:"), raw.slice(0, 200));
      transcript.push({ role: "system", content: 'Your last response was not valid JSON. Respond with ONLY a single JSON object. Example: {"step":"THINK","content":"..."}' });
      continue;
    }

    transcript.push({ role: "assistant", content: JSON.stringify(parsed) });

    // ── Handle step ──
    const { step, content, tool_name, tool_args } = parsed;

    if (step === "THINK") {
      console.log(clr(C.yellow, "💭 THINK: ") + content);

    } else if (step === "TOOL") {
      console.log(clr(C.magenta, "🔧 TOOL: ") + clr(C.bold, tool_name || "???"));

      let observeContent;

      if (!tool_name || !TOOL_MAP[tool_name]) {
        observeContent = `Tool "${tool_name}" not found. Available: ${Object.keys(TOOL_MAP).join(", ")}`;
        console.log(clr(C.red, `   ⚠️  ${observeContent}`));
      } else {
        try {
          const args = tool_args ?? {};
          const result = await TOOL_MAP[tool_name](args);
          observeContent = typeof result === "string" ? result : JSON.stringify(result, null, 2);
        } catch (e) {
          observeContent = `Tool error: ${e.message}`;
        }
      }

      const preview = observeContent.slice(0, 300) + (observeContent.length > 300 ? "\u2026" : "");
      console.log(clr(C.green, "   📋 OBSERVE: ") + clr(C.dim, preview));
      // Truncate what goes INTO the transcript to avoid 413 on Groq/OpenRouter
      const truncated = observeContent.length > 2000
        ? observeContent.slice(0, 2000) + "\n...[truncated]"
        : observeContent;
      transcript.push({ role: "observe", content: truncated });

    } else if (step === "OUTPUT") {
      console.log(clr(C.green, "\n✅ OUTPUT:\n") + clr(C.bold, content) + "\n");
      break;

    } else {
      // Nudge for unknown step
      transcript.push({ role: "system", content: `Unknown step "${step}". Use THINK, TOOL, or OUTPUT.` });
    }
  }

  if (iteration >= MAX_ITERATIONS) {
    console.log(clr(C.yellow, "\n⚠️  Max iterations reached. Agent stopped.\n"));
  }
}

// ─── CLI Interface ────────────────────────────────────────────────────────────

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function ask(prompt) {
  return new Promise((resolve) => rl.question(prompt, resolve));
}

// ─── Main Entry Point ─────────────────────────────────────────────────────────

async function main() {
  console.log("\x1b[36m\x1b[1m");
  console.log("╔══════════════════════════════════════════════════════════╗");
  console.log(`║   🚀  Scaler CLI Agent  ·  ${activeProviders.slice(0, 27).padEnd(28)}║`);
  console.log("║   ReAct Loop : THINK → TOOL → OBSERVE → OUTPUT           ║");
  console.log("╚══════════════════════════════════════════════════════════╝");
  console.log("\x1b[0m");
  console.log("\x1b[90mYou can ask me anything! Examples:\x1b[0m");
  console.log('\x1b[2m  • "What\'s the weather in Mumbai?"\x1b[0m');
  console.log('\x1b[2m  • "Show me GitHub info for torvalds"\x1b[0m');
  console.log('\x1b[2m  • "Create a todo app with HTML, CSS and JS"\x1b[0m');
  console.log('\x1b[2m  • "Clone the Scaler Academy website with Header, Hero and Footer"\x1b[0m');
  console.log('\x1b[2m  • "List files in this directory"\x1b[0m');
  console.log('\x1b[90m\nType "exit" or "quit" to close.\x1b[0m\n');

  while (true) {
    const userInput = await ask("\x1b[36m\x1b[1mYou:\x1b[0m ");
    const trimmed = userInput.trim();
    if (!trimmed) continue;
    if (["exit", "quit", "q", "bye"].includes(trimmed.toLowerCase())) {
      console.log(clr(C.cyan, "\nBye! 👋\n"));
      rl.close();
      break;
    }
    await runAgent(trimmed);
  }
}

main().catch((err) => {
  console.error(clr(C.red, `\nFatal error: ${err.message}`));
  process.exit(1);
});