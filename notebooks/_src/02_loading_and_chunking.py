# %% [markdown]
# # 02 · Loading and chunking, by hand
#
# Notebook 01 started from eight tidy passages. Real documents are PDFs with tables, Markdown with
# headers, HTML with navigation bars, and spreadsheets. Before we can retrieve anything we have to
# **load** them into text and **chunk** them into retrieval-sized pieces.
#
# **In this notebook you will:**
# - write loaders for PDF, Markdown, HTML and CSV, and look at the messy text they produce,
# - compare three chunking strategies: fixed-size, recursive and header-aware,
# - see why chunks overlap,
# - attach metadata to chunks, and see how a short "context prefix" helps retrieval.

# %%
# <<SETUP>>

# %%
# Setup 2/3: nothing to prepare. Everything here runs locally on the files in data/corpus.

# %%
# Setup 3/3: start the cost meter at zero
from raglab.llm import cost

cost.reset()

# %% [markdown]
# ## 1. Loading: every format is messy in its own way
#
# A loader turns a file into one or more **Documents**: `page_content` (text) plus `metadata`
# (where it came from). `raglab.loaders` has one small function per format. We borrow LangChain's
# `Document` class only as a container; it's a plain data class.

# %%
from raglab import loaders

corpus = settings.corpus_dir
pdf_pages = loaders.load_pdf(corpus / "hotel_profiles.pdf")
print(f"{len(pdf_pages)} pages")
print(pdf_pages[1].metadata)
print(pdf_pages[1].page_content[:900])

# %% [markdown]
# **PDF:** text comes out page by page, and the line breaks are wherever the PDF wrapped the line,
# not where sentences end. The amenities **table** is flattened into lines like
# `Swimming pool Yes`; the column structure is gone. A page also holds more than one hotel.

# %%
faq = loaders.load_html(corpus / "faq.html")[0]
print(faq.page_content[:400])

# %% [markdown]
# **HTML:** we dropped scripts and styles, but the navigation bar (`Home · Hotels · Offers`) and the
# footer are still there. Boilerplate like this ends up in chunks and in search results unless you
# remove it. We keep the raw HTML in `metadata["raw_html"]` because its `<h1>`/`<h2>` structure
# is useful for chunking (notebook 04).

# %%
rates = loaders.load_csv(corpus / "room_rates.csv")
print(f"{len(rates)} rows; first row as a document:\n{rates[0].page_content}")

# %% [markdown]
# **CSV:** each row becomes a tiny document. That works for lookups ("Suite at Token Kyoto in high
# season?") but it's a poor fit for questions like "cheapest suite in Asia?", which need arithmetic
# across rows. Notebook 12 gives that job to a proper tool instead.
#
# `load_corpus()` loads every file in one go:

# %%
docs = loaders.load_corpus()
import collections

collections.Counter(d.metadata["source_file"] for d in docs)

# %% [markdown]
# A quick sanity check worth doing on any new corpus: did accented characters survive? (Text files
# must be read as UTF-8; PDF extraction depends on the PDF's fonts.)

# %%
for d in docs:
    line = next((ln for ln in d.page_content.splitlines() if "Reykjav" in ln), None)
    if line:
        print(f"{d.metadata['source_file']:<28}{line.strip()[:70]}")

# %% [markdown]
# ## 2. Why chunk at all?
#
# One embedding per *document* would squeeze a whole dining guide (eight restaurants) into a
# single vector. A question about one restaurant would then be compared against an average of all
# eight. Smaller chunks give sharper vectors, and they keep the prompt short.
#
# But chunks that are *too* small lose context ("It opens at 7 PM." Which restaurant?). Chunking
# is a trade-off, and the strategy matters.

# %%
from raglab.display import show_diagram

show_diagram("chunking_strategies")

# %% [markdown]
# We'll compare the three strategies on the Markdown dining guide.

# %%
from raglab import chunking

dining = loaders.load_markdown(corpus / "dining_guide.md")
fixed = chunking.chunk_fixed(dining, size=500)
recursive = chunking.chunk_recursive(dining, size=500, overlap=100)
by_header = chunking.chunk_markdown_headers(dining)
print(f"fixed: {len(fixed)} chunks | recursive: {len(recursive)} | header-aware: {len(by_header)}")

# %% [markdown]
# ### Fixed-size: cut every N characters
#
# Look at where the first chunk ends and the second begins:

# %%
print(repr(fixed[0].page_content[-120:]))
print(repr(fixed[1].page_content[:120]))

# %% [markdown]
# The cut lands wherever character 500 falls, often mid-word or mid-sentence, so a fact can be
# split across two chunks and neither chunk contains it whole.
#
# ### Recursive: split at natural boundaries
#
# `chunk_recursive` tries the biggest boundary first (blank line), then a line break, then a
# sentence end (`". "`), then a space. Only pieces still too long are split further. Then it packs
# the pieces back together up to `size` characters.

# %%
for chunk in recursive[:2]:
    print(repr(chunk.page_content[:160]), "…", repr(chunk.page_content[-80:]), "\n")

# %% [markdown]
# Chunks now start and end at line or paragraph boundaries.
#
# ### Header-aware: one section per chunk
#
# Markdown already tells us where topics change: the headers. `chunk_markdown_headers` starts a new
# chunk at every `##` and `###`, keeps the header in the text, and records the header path in metadata.

# %%
from raglab.display import show_chunks

show_chunks(by_header, n=2, width=250)

# %% [markdown]
# Each chunk is exactly one restaurant, and `section` / `subsection` say which hotel it belongs to.
#
# ### Chunk sizes compared

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(
    [[len(c.page_content) for c in s] for s in (fixed, recursive, by_header)],
    bins=12,
    label=["fixed (500)", "recursive (500, overlap 100)", "header-aware"],
)
ax.set_xlabel("chunk length (characters)")
ax.set_ylabel("chunks")
ax.legend()
plt.show()

# %% [markdown]
# Fixed chunks are all the same size by construction. Header-aware chunks follow the content:
# one per restaurant, of whatever length that restaurant's entry happens to be. On a document with a
# very long section you'd combine both: split by headers first, then split any oversized section recursively.

# %% [markdown]
# ## 3. Overlap: don't lose facts at the seams
#
# With overlap, the start of each chunk repeats the end of the previous one, so a sentence near a
# boundary appears whole in at least one chunk. Let's chunk the first PDF page with
# `size=500, overlap=150`:

# %%
pdf_chunks = chunking.chunk_recursive(pdf_pages[:1], size=500, overlap=150)
first, second = pdf_chunks[0].page_content, pdf_chunks[1].page_content
shared = next(
    (second[:n] for n in range(min(len(first), len(second)), 0, -1) if first.endswith(second[:n])), ""
)
print(f"{len(shared)} characters shared between chunk 1 and chunk 2:")
print(repr(shared))

# %% [markdown]
# Our recursive chunker (like LangChain's) carries over **whole pieces**: here, whole lines of the
# PDF. If the pieces are bigger than the overlap budget, nothing is carried. Check
# `recursive[0]` and `recursive[1]` from the dining guide: its pieces are paragraphs longer than
# 100 characters, so those chunks share nothing. Overlap is a *maximum*, not a guarantee.
#
# Overlap costs a little storage and some duplicate text in prompts. A common starting point is
# 10–20% of the chunk size.

# %% [markdown]
# ## 4. Metadata: what is this chunk about?
#
# `enrich_metadata` adds a `doc_type` (from the file name) and, when a chunk is about exactly one
# hotel, that hotel's facts: city, continent, stars, price, pool, spa. Later notebooks filter on
# these fields ("hotels in Asia with a pool").
#
# Let's split the header chunks further into small pieces, like you might for a long document:

# %%
small = chunking.enrich_metadata(chunking.chunk_recursive(by_header, size=250, overlap=50))
orphans = [c for c in small if c.metadata.get("hotel_name") and c.metadata["hotel_name"] not in c.page_content]
print(f"{len(small)} small chunks; {len(orphans)} of them never mention their hotel in the text")
show_chunks(orphans, n=1, width=300)

# %% [markdown]
# That chunk is about one hotel's restaurant, but its text never says which hotel; only the
# header metadata knows. A question such as "What is the signature dish at Token Lisbon?" has
# nothing to match on.
#
# **Context prefix:** `add_context_prefix` prepends `"<hotel> — <doc type>"` to such chunks before
# they are embedded. Let's measure the effect with the similarity we built in notebook 01:

# %%
from raglab import llm, scratch

orphan = orphans[0]
with_prefix = chunking.add_context_prefix([orphan])[0]
question = f"What is the signature dish at {orphan.metadata['hotel_name']}?"
vectors = llm.embed([orphan.page_content, with_prefix.page_content])
q = llm.embed_one(question)
plain_score, prefixed_score = scratch.cosine_sim(q, vectors)
print(question)
print(f"similarity without prefix: {plain_score:.3f}")
print(f"similarity with prefix:    {prefixed_score:.3f}")
print("\nprefixed chunk starts with:", repr(with_prefix.page_content[:60]))

# %% [markdown]
# A few words of context move the chunk closer to the questions it should answer. Notebook 04
# applies this prefix to every chunk in the index.
#
# ## Your turn
#
# 1. Run `chunk_fixed(dining, size=200)` and find a chunk that cuts a restaurant's hours or price
#    in half. What question would now be impossible to answer from a single chunk?
# 2. Chunk the PDF pages with `chunk_recursive(pdf_pages, 800, 150)` and run `enrich_metadata`.
#    How many chunks get a `hotel_name`? Why do the others not get one?
# 3. Try `chunk_markdown_headers(dining, levels=(2,))`. How many chunks do you get now, and what's in them?
#
# <details><summary>Hints</summary>
#
# 2. A PDF page holds two hotels, so an 800-character chunk often straddles both. `enrich_metadata`
#    only assigns a hotel when exactly one is named. Structure-aware splitting (or smaller chunks)
#    fixes this, at the cost of more chunks.
# 3. Splitting only at `##` gives one chunk per *hotel* (each containing all of its restaurants),
#    a middle ground between the two strategies.
# </details>
#
# ## Key takeaways
#
# - Loaders produce imperfect text: PDF tables flatten, HTML carries boilerplate, CSV rows lack context.
# - Recursive chunking respects sentences; header-aware chunking respects topics. Overlap protects seams.
# - Metadata (doc type, hotel, city…) enables filtering, and a context prefix makes small chunks findable.
#
# **Added to raglab:** `loaders.load_pdf / load_markdown / load_html / load_csv / load_corpus`,
# `chunking.chunk_fixed / chunk_recursive / chunk_markdown_headers / enrich_metadata / add_context_prefix`.

# %%
cost.report()
