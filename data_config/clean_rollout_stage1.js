const fs = require("fs");
const crypto = require("crypto");

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  throw new Error("Usage: node clean_rollout_stage1.js INPUT.json OUTPUT.json");
}

const EXPECTED = Object.freeze({
  inputRows: 12847,
  blankProblems: 1,
  duplicateRows: 4,
  optionRowsNormalized: 126,
  outputRows: 12842,
});

const sha256 = (buffer) => crypto.createHash("sha256").update(buffer).digest("hex");
const raw = fs.readFileSync(inputPath);
const rows = JSON.parse(raw.toString("utf8"));

if (!Array.isArray(rows) || rows.length !== EXPECTED.inputRows) {
  throw new Error(`Unexpected input row count: ${Array.isArray(rows) ? rows.length : "not-an-array"}`);
}

const choiceTypes = new Set(["multiple choice", "emer_ov_mc"]);
const optionLabel = /^\s*([A-Z])\s*[.\):：]\s*(.*?)\s*$/s;
const labelForIndex = (index) => String.fromCharCode("A".charCodeAt(0) + index);

let blankProblemsRemoved = 0;
let duplicateRowsRemoved = 0;
let exactDuplicateRowsObserved = 0;
let optionRowsNormalized = 0;
let unlabeledRowsNormalized = 0;
let artifactRowsNormalized = 0;
let artifactOptionsRemoved = 0;

const seenExactRows = new Set();
const seenSemanticRows = new Set();
const cleaned = [];

for (const original of rows) {
  if (!String(original.problem ?? "").trim()) {
    blankProblemsRemoved += 1;
    continue;
  }

  const exactRowKey = JSON.stringify(original);
  if (seenExactRows.has(exactRowKey)) {
    exactDuplicateRowsObserved += 1;
  } else {
    seenExactRows.add(exactRowKey);
  }

  // Rollout concatenation produced repeated training examples whose metadata or
  // rewritten rationale can differ. They are the same supervised/RL item when
  // media path, prompt, and ground-truth answer are identical.
  const semanticRowKey = JSON.stringify([
    String(original.path ?? ""),
    String(original.problem ?? ""),
    String(original.answer ?? ""),
  ]);
  if (seenSemanticRows.has(semanticRowKey)) {
    duplicateRowsRemoved += 1;
    continue;
  }
  seenSemanticRows.add(semanticRowKey);

  const row = { ...original };
  if (choiceTypes.has(row.problem_type) && Array.isArray(row.options) && row.options.length > 0) {
    const parsed = row.options.map((option) => optionLabel.exec(String(option)));
    const parsedCount = parsed.filter(Boolean).length;

    if (parsedCount === 0) {
      row.options = row.options.map(
        (option, index) => `${labelForIndex(index)}. ${String(option).trim()}`
      );
      optionRowsNormalized += 1;
      unlabeledRowsNormalized += 1;
    } else if (parsedCount !== row.options.length) {
      const validOptions = parsed.filter(Boolean).map((match) => match[2].trim());
      artifactOptionsRemoved += row.options.length - validOptions.length;
      row.options = validOptions.map(
        (option, index) => `${labelForIndex(index)}. ${option}`
      );
      optionRowsNormalized += 1;
      artifactRowsNormalized += 1;
    }
  }

  cleaned.push(row);
}

const answerOutOfRange = [];
for (let index = 0; index < cleaned.length; index += 1) {
  const row = cleaned[index];
  if (!choiceTypes.has(row.problem_type) || !Array.isArray(row.options)) continue;
  const answer = String(row.answer ?? "").trim();
  if (/^[A-Z]$/.test(answer)) {
    const answerIndex = answer.charCodeAt(0) - "A".charCodeAt(0);
    if (answerIndex < 0 || answerIndex >= row.options.length) {
      answerOutOfRange.push({ index, answer, optionCount: row.options.length });
    }
  }
}

const actual = {
  inputRows: rows.length,
  blankProblems: blankProblemsRemoved,
  duplicateRows: duplicateRowsRemoved,
  optionRowsNormalized,
  outputRows: cleaned.length,
};

for (const [name, expected] of Object.entries(EXPECTED)) {
  if (actual[name] !== expected) {
    throw new Error(`${name}: expected ${expected}, observed ${actual[name]}`);
  }
}
if (answerOutOfRange.length > 0) {
  throw new Error(`Choice answers out of range after cleaning: ${JSON.stringify(answerOutOfRange.slice(0, 10))}`);
}

const output = Buffer.from(`${JSON.stringify(cleaned, null, 2)}\n`, "utf8");
fs.writeFileSync(outputPath, output);

console.log(JSON.stringify({
  inputPath,
  outputPath,
  inputSha256: sha256(raw),
  outputSha256: sha256(output),
  inputRows: rows.length,
  blankProblemsRemoved,
  duplicateRowsRemoved,
  exactDuplicateRowsObserved,
  optionRowsNormalized,
  unlabeledRowsNormalized,
  artifactRowsNormalized,
  artifactOptionsRemoved,
  answerOutOfRange: answerOutOfRange.length,
  outputRows: cleaned.length,
}, null, 2));
