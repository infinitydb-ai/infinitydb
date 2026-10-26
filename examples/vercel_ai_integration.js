/**
 * InfinityDB Drop-in Memory Middleware for Vercel AI SDK.
 *
 * Enriches prompts and chat completions with sub-millisecond local memories,
 * user persona facts, and corporate policies before sending to OpenAI/Anthropic.
 */

"use strict";

const { InfinityDB } = require("../infinitydb-js/index.js");

class InfinityDBVercelMemory {
  constructor(options = {}) {
    this.db = options.db || new InfinityDB();
    this.userId = options.userId || "default_user";
    this.limit = options.limit || 5;
  }

  /**
   * Save a dialogue exchange into persistent L0-L6 memory.
   */
  async saveContext(prompt, response, metadata = {}) {
    const text = `User: ${prompt}\nAssistant: ${response}`;
    return this.db.add(text, this.userId, metadata.sessionId || null, 0.9);
  }

  /**
   * Recall memories and construct enriched prompt for Vercel AI SDK.
   */
  async enrichPrompt(prompt) {
    const recallRes = this.db.recall(prompt, this.limit, this.userId);
    const facts = (recallRes.results || []).map((r) => r.fact);
    const persona = this.db.getPersona(this.userId);

    let contextBlock = "";
    if (facts.length > 0) {
      contextBlock += `Recalled User Facts:\n${facts.map((f) => `- ${f}`).join("\n")}\n\n`;
    }
    if (Object.keys(persona).length > 0) {
      contextBlock += `User Persona:\n${Object.entries(persona).map(([k, v]) => `- ${k}: ${v}`).join("\n")}\n\n`;
    }

    if (!contextBlock) {
      return prompt;
    }

    return `${contextBlock}Current User Request: ${prompt}`;
  }

  /**
   * Vercel AI SDK transform middleware adapter.
   */
  createMiddleware() {
    return {
      wrapGenerate: async ({ doGenerate, params }) => {
        const originalPrompt = params.prompt || "";
        const enrichedPrompt = await this.enrichPrompt(originalPrompt);
        params.prompt = enrichedPrompt;

        const result = await doGenerate();
        if (result.text) {
          await this.saveContext(originalPrompt, result.text);
        }
        return result;
      },
    };
  }
}

// Verification demo
async function runDemo() {
  console.log("[*] Initializing InfinityDB Vercel AI SDK Middleware...");
  const memory = new InfinityDBVercelMemory({ userId: "alex_dev" });

  // Simulate past interactions
  await memory.saveContext(
    "I prefer Next.js 15 with Tailwind CSS for web apps.",
    "Got it, Next.js 15 and Tailwind CSS set as default."
  );
  await memory.saveContext(
    "Never use Redux; I prefer Zustand for state management.",
    "Noted, Zustand will be used for state management."
  );

  // New incoming prompt
  const incomingPrompt = "What framework and state library should I initialize?";
  const enriched = await memory.enrichPrompt(incomingPrompt);

  console.log("\n--- Enriched Vercel AI SDK Prompt ---");
  console.log(enriched);

  console.log("\n--- Stats ---");
  console.log(memory.db.stats());
}

if (require.main === module) {
  runDemo();
}

module.exports = { InfinityDBVercelMemory };
