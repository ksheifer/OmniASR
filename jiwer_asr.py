from jiwer import wer, cer, mer, wil, wip, process_words, process_characters

hyp_path = "/Users/karinasheifer/Documents/UCSB/ASR/cluster_testing/revenge_vad.txt"
ref_path = "/Users/karinasheifer/Documents/UCSB/ASR/cluster_testing/revenge_truth.txt"
out_path = "/Users/karinasheifer/Documents/UCSB/ASR/cluster_testing/revenge_vad_jiwer.txt"

with open(hyp_path, "r", encoding="utf-8") as f:
    hypothesis = f.read().strip()

with open(ref_path, "r", encoding="utf-8") as f:
    reference = f.read().strip()

# ================= CORE METRICS =================
word_error_rate = wer(reference, hypothesis)
char_error_rate = cer(reference, hypothesis)
match_error_rate = mer(reference, hypothesis)
word_info_lost = wil(reference, hypothesis)
word_info_preserved = wip(reference, hypothesis)

# ================= WORD STATS =================
word_stats = process_words(reference, hypothesis)
total_ref_words = word_stats.hits + word_stats.substitutions + word_stats.deletions
total_hyp_words = word_stats.hits + word_stats.substitutions + word_stats.insertions

# Разворачиваем списки слов
ref_words = [w for sent in word_stats.references for w in sent]
hyp_words = [w for sent in word_stats.hypotheses for w in sent]

# ================= CHAR STATS =================
char_stats = process_characters(reference, hypothesis)
total_ref_chars = char_stats.hits + char_stats.substitutions + char_stats.deletions

# ================= HYPHENATION METRICS =================
ref_hyph_total = 0
hyp_hyph_total = 0
correct_hyph = 0

missing_examples = []
extra_examples = []
correct_examples = []

for sentence_chunks in word_stats.alignments:
    for chunk in sentence_chunks:
        ref_segment = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
        hyp_segment = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]

        if chunk.type in ("equal", "substitute"):
            for r_word, h_word in zip(ref_segment, hyp_segment):
                r_h = r_word.count("-")
                h_h = h_word.count("-")

                ref_hyph_total += r_h
                hyp_hyph_total += h_h

                if r_h > 0 and h_h > 0:
                    correct_hyph += min(r_h, h_h)
                    correct_examples.append(f"{r_word}  →  {h_word}")

                if r_h > h_h:
                    missing_examples.append(f"{r_word}  →  {h_word}")

                if h_h > r_h:
                    extra_examples.append(f"{r_word}  →  {h_word}")

        elif chunk.type == "delete":
            for r_word in ref_segment:
                r_h = r_word.count("-")
                ref_hyph_total += r_h
                if r_h > 0:
                    missing_examples.append(f"{r_word}  →  Ø")

        elif chunk.type == "insert":
            for h_word in hyp_segment:
                h_h = h_word.count("-")
                hyp_hyph_total += h_h
                if h_h > 0:
                    extra_examples.append(f"Ø  →  {h_word}")

missing_hyph = ref_hyph_total - correct_hyph
extra_hyph = hyp_hyph_total - correct_hyph

hyph_recall = correct_hyph / ref_hyph_total if ref_hyph_total else 1.0
hyph_precision = correct_hyph / hyp_hyph_total if hyp_hyph_total else 1.0


# ================= REPORT =================
report = f"""
=== CORE METRICS ===
WER: {word_error_rate:.3f}
CER: {char_error_rate:.3f}
MER: {match_error_rate:.3f}
WIL: {word_info_lost:.3f}
WIP: {word_info_preserved:.3f}

=== WORD-LEVEL DETAILS ===
Reference words:  {total_ref_words}
Hypothesis words: {total_hyp_words}
Hits:             {word_stats.hits}
Substitutions:    {word_stats.substitutions}
Deletions:        {word_stats.deletions}
Insertions:       {word_stats.insertions}

=== CHARACTER-LEVEL DETAILS ===
Reference characters: {total_ref_chars}
Hits:                 {char_stats.hits}
Substitutions:        {char_stats.substitutions}
Deletions:            {char_stats.deletions}
Insertions:           {char_stats.insertions}

=== HYPHENATION ANALYSIS ===
Reference hyphens:    {ref_hyph_total}
Hypothesis hyphens:   {hyp_hyph_total}
Correct hyphenations: {correct_hyph}
Missing hyphens:      {missing_hyph}
Extra hyphens:        {extra_hyph}

Hyphen Recall:        {hyph_recall:.3f}
Hyphen Precision:     {hyph_precision:.3f}

--- Correct Hyphenation Examples ---
{chr(10).join(correct_examples[:30])}

--- Missing Hyphen Examples (lost in ASR) ---
{chr(10).join(missing_examples[:30])}

--- Extra Hyphen Examples (added by ASR) ---
{chr(10).join(extra_examples[:30])}

"""

with open(out_path, "w", encoding="utf-8") as f:
    f.write(report)

print("Analysis saved to jiwer.txt with full hyphenation diagnostics")
