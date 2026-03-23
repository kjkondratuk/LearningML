# Module 01: Paper Reading Methodology

## The Three-Pass Method (Keshav 2007)

### Pass 1: Survey (5-10 minutes)
- Read title, abstract, introduction, section headings, conclusion
- Glance at figures and tables
- Answer: What is the problem? What is the approach? What are the results?

### Pass 2: Comprehend (1-2 hours)
- Read the full paper, skipping proofs and dense math on first read
- Annotate key claims, equations, and experimental details
- Identify the core contribution vs incremental improvements
- Note what you do not understand for Pass 3

### Pass 3: Reproduce (4-8 hours)
- Reconstruct the method from scratch
- Verify key equations by deriving them yourself
- Identify all assumptions and where they might break
- Find the open questions and limitations

## Evaluating a Paper

### Experimental Methodology
- Are baselines fair? (Same compute budget, data, tuning effort)
- Are ablations thorough? (Does each component contribute?)
- Is statistical significance reported? (Error bars, multiple seeds)

### Identifying Limitations
- What assumptions does the method make?
- Under what conditions would it fail?
- What is the computational cost?

### Finding Open Questions
- What extensions are natural?
- What related problems could this approach solve?
- Where is the theory incomplete?

## How to Take Notes

Use the reading template in `reading_template.md` to structure your notes. The template
covers: problem statement, prior work, key insight, method summary, experimental setup,
results, ablations, limitations, and your ideas for extensions.
