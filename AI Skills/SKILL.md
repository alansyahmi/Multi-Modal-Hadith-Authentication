---
name: ghost
description: Humanize AI-generated text to reduce AI detection and improve originality. Use this skill whenever you need to transform AI-written content into natural, human-sounding text that passes Turnitin and other AI detection tools. Trigger when you have AI-generated text that needs to sound more conversational, original, and authentic — whether it's for academic proposals, essays, articles, or any document where AI detection is a concern. Also use when you want to refactor predictable or formulaic writing into more varied and engaging prose. Input can be plain text excerpts or full Word documents (.docx files).
compatibility: 
  - Requires: python-docx library for Word document processing
  - Optional: Access to the user's .docx files
---

# Ghost: Humanize AI-Generated Text

## What This Skill Does

The Ghost skill transforms AI-generated or AI-assisted text into natural, human-sounding writing. It specifically targets the patterns that AI detection tools (like Turnitin's AI detector) flag:

- **Repetitive sentence structures** → varied rhythm and length
- **Formal, generic phrasing** → casual, conversational tone
- **Common AI phrases** → original, specific wording
- **Predictable transitions** → natural, varied connectors
- **Lack of personal voice** → authentic perspective and examples

This is particularly useful for academic work, FYP proposals, and other submissions where both plagiarism detection (<20% similarity) and AI detection matter.

## How to Use This Skill

### Input Format

You can provide:
1. **Plain text snippets** — Paste the AI-generated text directly
2. **Word document (.docx)** — The skill will process and return a humanized version
3. **Mixed content** — Multiple sections to humanize at once

### The Humanization Process

When you use this skill, Claude will:

1. **Analyze the text** for AI markers:
   - Overly formal or generic phrases
   - Repetitive sentence patterns
   - Lack of personal examples or perspective
   - Predictable transitions and connectors

2. **Restructure and rewrite** with focus on:
   - **Varied sentence structure** — Mix short, punchy sentences with longer, complex ones
   - **Casual phrasing** — Add contractions, conversational language, and natural connectors
   - **Specific examples** — Include concrete details instead of generic statements
   - **Original wording** — Avoid common AI phrases and clichés
   - **Personal voice** — Inject perspective, reasoning, and authentic tone

3. **Preserve meaning** — All original ideas and information remain intact; only the expression changes

4. **Output the result** in your requested format (plain text or .docx)

## Step-by-Step Instructions

### For Plain Text

1. Share the AI-generated text you want humanized
2. Optionally specify any constraints:
   - Maintain formal/academic tone while sounding more human
   - Keep it concise or expand with examples
   - Focus on specific sections (introduction, methodology, conclusion)
3. Claude will rewrite it, explaining the key changes made
4. Review and request adjustments if needed

### For Word Documents

1. Upload your .docx file
2. Claude will:
   - Extract the text
   - Humanize key sections (or all sections)
   - Rebuild the document with your formatting intact
   - Return a new .docx file ready to use
3. Download and review in Word

## Why This Helps with Turnitin

Turnitin uses multiple detection mechanisms:

- **Similarity checking** — Compares against known sources. Humanization doesn't directly help here (use proper citations instead), but varied phrasing reduces false matches.
- **AI detection** — Analyzes writing patterns, vocabulary, predictability, and tone consistency. Humanization directly addresses these:
  - ✅ Varied sentence structure breaks pattern detection
  - ✅ Casual phrasing and contractions signal human writing
  - ✅ Original wording avoids AI phrase libraries
  - ✅ Personal voice and specific examples are hard to fake algorithmically

## Best Practices

1. **Start with a solid foundation** — Humanization works best on text that's already well-structured and accurate. Use AI to draft, then use Ghost to humanize.

2. **Keep your own voice** — The skill enhances authenticity, but the core ideas and voice should be yours. Use it to refine, not replace, your thinking.

3. **Review carefully** — Always read the humanized output. Make sure it still captures your original intent and doesn't accidentally introduce errors.

4. **Use strategically** — Focus on sections most likely to trigger AI detection:
   - Introduction and conclusion (often generic)
   - Methodology descriptions
   - Literature review summaries
   - Any sections generated wholesale by AI

5. **Combine with other practices**:
   - Proper citations (reduces similarity score)
   - Your own original research and examples
   - Personal reflection and analysis
   - Multiple drafts and revisions

## Example Transformations

**Before (AI-generated):**
> "Artificial intelligence has emerged as a transformative technology that has significant implications for various industries. The advancement of machine learning algorithms has enabled organizations to leverage data-driven insights for improved decision-making processes. Furthermore, the integration of AI solutions has demonstrated considerable potential in enhancing operational efficiency and reducing costs."

**After (Humanized):**
> "AI is genuinely changing how businesses operate. What's interesting is how machine learning actually lets companies dig into their data and make smarter choices. I've seen firsthand how AI integration helps teams work faster and cut unnecessary spending — which matters a lot when budgets are tight."

---

## Technical Details

**Supported File Types:**
- `.docx` (Microsoft Word) — full document processing
- Plain text (copy/paste) — immediate humanization
- PDF → Word conversion (available via additional tools)

**Output:**
- Humanized text in your chosen format
- Optional analysis of changes made
- Suggestions for further refinement

**Limitations:**
- Works best on paragraphs and longer passages (short fragments may be unchanged)
- Cannot guarantee 100% undetection by any specific tool, but significantly reduces AI markers
- Requires your review to ensure accuracy and personal voice are maintained

---

## Getting Started

Simply share your AI-generated text or .docx file with a message like:

- "Humanize this text: [paste content]"
- "Upload: FYP_proposal_draft.docx and make it sound more human"
- "Can you make this methodology section less AI-sounding?"

Claude will take it from there!
